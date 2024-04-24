from typing import Any, Dict, List, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# TODO (stao): we need to cut the meshes of all the cabinets in this dataset for gpu sim, not registering task for now
@register_env("OpenCabinetDrawer-v1", max_episode_steps=100)
class OpenCabinetDrawerEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["fetch"]
    agent: Union[Fetch]
    handle_types = ["prismatic"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    min_open_frac = 0.75

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        # self.all_model_ids = np.array(["1004", "1004"])
        if reconfiguration_freq is None:
            # if not user set, we pick a number
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=10,
            gpu_memory_cfg=GPUMemoryConfig(
                max_rigid_contact_count=2**23, max_rigid_patch_count=2**21
            ),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-1.8, -1.3, 1.8], target=[-0.3, 0.5, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.ground = build_ground(self._scene)
        self._load_cabinets(self.handle_types)

        from mani_skill.agents.robots.fetch import FETCH_UNIQUE_COLLISION_BIT

        # TODO (stao) (arth): is there a better way to model robots in sim. This feels very unintuitive.
        for obj in self.ground._objs:
            for cs in obj.find_component_by_type(
                sapien.physx.PhysxRigidStaticComponent
            ).get_collision_shapes():
                cg = cs.get_collision_groups()
                cg[2] |= FETCH_UNIQUE_COLLISION_BIT
                cg[2] |= 1 << 29  # make ground ignore collisions with the cabinets
                cs.set_collision_groups(cg)

    def _load_cabinets(self, joint_types: List[str]):
        rand_idx = self._episode_rng.permutation(np.arange(0, len(self.all_model_ids)))
        model_ids = self.all_model_ids[rand_idx]
        model_ids = np.concatenate(
            [model_ids] * np.ceil(self.num_envs / len(self.all_model_ids)).astype(int)
        )[: self.num_envs]
        link_ids = self._episode_rng.randint(0, 2**31, size=len(model_ids))

        self._cabinets = []
        self.cabinet_zs = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            cabinet_builder = articulations.get_articulation_builder(
                self._scene, f"partnet-mobility:{model_id}"
            )
            cabinet_builder.set_scene_idxs(scene_idxs=[i])
            cabinet = cabinet_builder.build(name=f"{model_id}-{i}")
            self._cabinets.append(cabinet)
            handle_links.append([])
            handle_links_meshes.append([])

            # TODO (stao): At the moment code for selecting semantic parts of articulations
            # is not very simple
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a handle
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, x: "handle" in x.name, mesh_name="handle"
                        )[0]
                    )

        # we can merge different articulations with different degrees of freedoms into a single view/object
        # allowing you to manage all of them under one object and retrieve data like qpos, pose, etc. all together
        # and with high performance. Note that some properties such as qpos and qlimits are now padded.
        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        # store the position of the handle mesh itself relative to the link it is apart of
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            )
        )

        self.handle_link_goal = actors.build_sphere(
            self._scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.handle_link_goal)

    def _after_reconfigure(self, options):
        # To spawn cabinets in the right place, we need to change their z position such that
        # the bottom of the cabinet sits at z=0 (the floor). Luckily the partnet mobility dataset is made such that
        # the negative of the lower z-bound of the collision mesh bounding box is the right value

        # this code is in _after_reconfigure since retrieving collision meshes requires the GPU to be initialized
        # which occurs after the initial reconfigure call (after self._load_scene() is called)
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs)

        # get the qmin qmax values of the joint corresponding to the selected links
        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: torch.Tensor = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        with torch.device(self.device):
            b = len(env_idx)
            xy = torch.zeros((b, 3))
            xy[:, 2] = self.cabinet_zs[env_idx]
            self.cabinet.set_pose(Pose.create_from_pq(p=xy))
            # the three lines here are necessary to update all link poses whenever qpos and root pose of articulation change
            # that way you can use the correct link poses as done below
            if physx.is_gpu_enabled():
                self._scene._gpu_apply_all()
                self._scene.px.gpu_update_articulation_kinematics()
                self._scene._gpu_fetch_all()
            # handle_link_positions is the position handle mesh itself
            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

            # close all the cabinets. We know beforehand that lower qlimit means "closed" for these assets.
            qlimits = self.cabinet.get_qlimits()  # [b, self.cabinet.max_dof, 2])
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])

            # NOTE (stao): This is a temporary work around for the issue where the cabinet drawers/doors might open
            # themselves on the first step. It's unclear why this happens on GPU sim only atm.
            if physx.is_gpu_enabled():
                self._scene._gpu_apply_all()
                self._scene.px.step()

            # initialize robot
            if self.robot_uids == "fetch":
                qpos = torch.tensor(
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -np.pi / 4,
                        0,
                        np.pi / 4,
                        0,
                        np.pi / 3,
                        0,
                        0.015,
                        0.015,
                    ]
                )
                qpos = qpos.repeat(b).reshape(b, -1)
                dist = randomization.uniform(1.6, 1.8, size=(b,))
                theta = randomization.uniform(0.9 * torch.pi, 1.1 * torch.pi, size=(b,))
                xy = torch.zeros((b, 2))
                xy[:, 0] += torch.cos(theta) * dist
                xy[:, 1] += torch.sin(theta) * dist
                qpos[:, :2] = xy
                noise_ori = randomization.uniform(
                    -0.05 * torch.pi, 0.05 * torch.pi, size=(b,)
                )
                ori = (theta - torch.pi) + noise_ori
                qpos[:, 2] = ori
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose())

    ### Useful properties ###

    def evaluate(self):
        # even though self.handle_link is a different link across different articulations
        # we can still fetch a joint that represents the parent joint of all those links
        # and easily get the qpos value.
        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        handle_link_pos = self.handle_link_positions()
        self.handle_link_goal.set_pose(Pose.create_from_pq(p=handle_link_pos))
        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
        return {
            "success": open_enough & link_is_static,
            "handle_link_pos": handle_link_pos,
            "open_enough": open_enough,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        amount_to_open_left = torch.div(
            self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
        )
        open_reward = 2 * (1 - amount_to_open_left)
        reaching_reward[
            amount_to_open_left < 0.999
        ] = 2  # if joint opens even a tiny bit, we don't need reach reward anymore
        # print(open_reward.shape)
        open_reward[info["open_enough"]] = 3  # give max reward here
        reward = reaching_reward + open_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("OpenCabinetDoor-v1", max_episode_steps=100)
class OpenCabinetDoorEnv(OpenCabinetDrawerEnv):
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"
    )
    handle_types = ["revolute", "revolute_unwrapped"]
