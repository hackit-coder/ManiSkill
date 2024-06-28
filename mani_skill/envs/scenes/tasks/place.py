from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
import sapien
import sapien.physx as physx
import trimesh

from mani_skill.envs.utils import randomization
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill import ASSET_DIR

from .planner import (
    TaskPlan,
    PlaceSubtask,
    PlaceSubtaskConfig,
)
from .subtask import SubtaskTrainEnv


@register_env("PlaceSubtaskTrain-v0", max_episode_steps=200)
class PlaceSubtaskTrainEnv(SubtaskTrainEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    place_cfg = PlaceSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        robot_init_qpos_noise=0.2,
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        **kwargs,
    ):

        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], PlaceSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {PlaceSubtask.__name__} long"

        self.subtask_cfg = self.place_cfg

        self.place_obj_ids = set()
        for tp in task_plans:
            self.place_obj_ids.add("-".join(tp.subtasks[0].obj_id.split("-")[:-1]))

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------
    # TODO (arth): maybe check that obj won't fall when noise is added
    # -------------------------------------------------------------------------------------------------

    def _after_reconfigure(self, options):

        self.grasping_spawns = dict()
        for obj_id in self.place_obj_ids:
            with open(
                ASSET_DIR
                / "scene_datasets/replica_cad_dataset/rearrange/grasp_poses"
                / f"{obj_id}_success_grasp_poses.pt",
                "rb",
            ) as spawns_fp:
                data = torch.load(spawns_fp)
                data["success_qpos"] = torch.tensor(
                    data["success_qpos"], device=self.device
                )
                data["success_obj_raw_pose_wrt_tcp"] = torch.tensor(
                    data["success_obj_raw_pose_wrt_tcp"], device=self.device
                )
                self.grasping_spawns[obj_id] = data

        return super()._after_reconfigure(options)

    def process_task_plan(self, sampled_subtask_lists: List[List[PlaceSubtask]]):

        self.current_place_obj_ids = [
            "-".join(subtask_list[0].obj_id.split("-")[:-1])
            for subtask_list in sampled_subtask_lists
        ]
        super().process_task_plan(sampled_subtask_lists)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            original_env_idx = env_idx.clone()
            init_success = torch.zeros(self.num_envs, dtype=torch.bool)

            super()._initialize_episode(env_idx, options)

            robot_init_p, robot_init_q, robot_init_qpos, obj_init_raw_pose_wrt_tcp = (
                self.agent.robot.pose.p.clone(),
                self.agent.robot.pose.q.clone(),
                self.agent.robot.qpos.clone(),
                self.subtask_objs[0].pose.raw_pose.clone(),
            )
            # keep going until no collisions
            while True:
                centers = self.subtask_goals[0].pose.p[env_idx, :2]
                navigable_positions = []
                for env_num, center in zip(env_idx, centers):
                    env_navigable_positions = self.scene_builder.navigable_positions[
                        env_num
                    ]
                    if isinstance(env_navigable_positions, trimesh.Trimesh):
                        env_navigable_positions = env_navigable_positions.vertices
                    positions = torch.tensor(env_navigable_positions)
                    navigable_positions.append(
                        positions[
                            torch.norm(positions - center, dim=1)
                            <= self.spawn_loc_radius
                        ]
                    )
                num_navigable_positions = torch.tensor(
                    [len(positions) for positions in navigable_positions]
                ).int()
                navigable_positions = pad_sequence(
                    navigable_positions, batch_first=True, padding_value=0
                ).float()

                positions_wrt_centers = navigable_positions - centers.unsqueeze(1)
                dists = torch.norm(positions_wrt_centers, dim=-1)

                rots = (
                    torch.sign(positions_wrt_centers[..., 1])
                    * torch.arccos(positions_wrt_centers[..., 0] / dists)
                    + torch.pi
                )
                rots %= 2 * torch.pi

                # ---------------------------------------------------
                # Sample grasp qpos and obj pose relative to tcp
                # ---------------------------------------------------
                qpos = robot_init_qpos.clone()
                obj_raw_pose_wrt_tcp = obj_init_raw_pose_wrt_tcp.clone()
                for env_num, obj_id in zip(
                    env_idx,
                    [
                        self.current_place_obj_ids[env_num]
                        for env_num in env_idx.tolist()
                    ],
                ):
                    spawns = self.grasping_spawns[obj_id]
                    spawn_num = torch.randint(
                        low=0, high=len(spawns["success_qpos"]), size=(1,)
                    ).item()
                    qpos[env_num] = spawns["success_qpos"][spawn_num]
                    obj_raw_pose_wrt_tcp[env_num] = spawns[
                        "success_obj_raw_pose_wrt_tcp"
                    ][spawn_num]
                # ---------------------------------------------------

                # ---------------------------------------------------
                # FIRST, check robot collision
                # ---------------------------------------------------
                if self.randomize_loc:
                    low = torch.zeros_like(num_navigable_positions)
                    high = num_navigable_positions
                    size = env_idx.size()
                    idxs: List[int] = (
                        (torch.randint(2**63 - 1, size=size) % (high - low) + low)
                        .int()
                        .tolist()
                    )
                    locs = torch.stack(
                        [
                            positions[i]
                            for positions, i in zip(navigable_positions, idxs)
                        ],
                        dim=0,
                    )
                    rots = torch.stack(
                        [rot[i] for rot, i in zip(rots, idxs)],
                        dim=0,
                    )
                else:
                    raise NotImplementedError()
                robot_pos = self.agent.robot.pose.p
                robot_pos[env_idx, :2] = locs
                self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

                qpos[env_idx, 2] = rots
                if self.randomize_base:
                    # base pos
                    robot_pos = self.agent.robot.pose.p
                    robot_pos[env_idx, :2] += torch.clamp(
                        torch.normal(0, 0.1, robot_pos[env_idx, :2].shape), -0.2, 0.2
                    ).to(self.device)
                    self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))
                    # base rot
                    qpos[env_idx, 2:3] += torch.clamp(
                        torch.normal(0, 0.25, qpos[env_idx, 2:3].shape), -0.5, 0.5
                    ).to(self.device)
                if self.randomize_arm:
                    rrqd = self.subtask_cfg.robot_init_qpos_noise
                    qpos[env_idx, 5:6] += torch.clamp(
                        torch.normal(0, rrqd / 2, qpos[env_idx, 5:6].shape), -rrqd, rrqd
                    ).to(self.device)
                    qpos[env_idx, 7:-2] += torch.clamp(
                        torch.normal(0, rrqd / 2, qpos[env_idx, 7:-2].shape),
                        -rrqd,
                        rrqd,
                    ).to(self.device)
                self.agent.reset(qpos)
                grasp_tcp_pose = self.agent.tcp_pose

                robot_init_p[env_idx] = self.agent.robot.pose.p[env_idx].clone()
                robot_init_q[env_idx] = self.agent.robot.pose.q[env_idx].clone()
                robot_init_qpos[env_idx] = self.agent.robot.qpos[env_idx].clone()

                if physx.is_gpu_enabled():
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()
                self.scene.step()

                robot_force = self.agent.robot.get_net_contact_forces(
                    self.force_articulation_link_ids
                ).norm(dim=-1)
                if physx.is_gpu_enabled():
                    robot_force = robot_force.sum(dim=-1)

                robot_spawn_fail = robot_force >= 1e-3

                # ---------------------------------------------------

                # ---------------------------------------------------
                # SECOND, check object collision
                # ---------------------------------------------------
                # NOTE: if collided, tcp pose will be wrong. However,
                #   if collided then init failed anyways, so that's ok

                self.scene_builder.initialize(original_env_idx, self.init_config_idxs)
                self.agent.robot.set_pose(Pose.create(sapien.Pose(p=[99999] * 3)))
                self.subtask_objs[0].set_pose(
                    grasp_tcp_pose * Pose.create(obj_raw_pose_wrt_tcp)
                )

                obj_init_raw_pose_wrt_tcp[env_idx] = obj_raw_pose_wrt_tcp[
                    env_idx
                ].clone()

                if physx.is_gpu_enabled():
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()
                self.scene.step()

                obj_force = self.subtask_objs[0].get_net_contact_forces().norm(dim=-1)
                obj_spawn_fail = obj_force >= 1e-3

                # ---------------------------------------------------

                # ---------------------------------------------------
                # FINALLY, check respawn criteria
                # ---------------------------------------------------
                # NOTE: if collided, tcp pose will be wrong. However,
                #   if collided then init failed anyways, so that's ok
                # ---------------------------------------------------

                init_success = (~robot_spawn_fail & ~obj_spawn_fail) | init_success
                env_idx = torch.where(~init_success)[0]

                self.scene_builder.initialize(original_env_idx, self.init_config_idxs)
                self.agent.reset(robot_init_qpos)
                self.agent.robot.set_pose(
                    Pose.create_from_pq(p=robot_init_p, q=robot_init_q)
                )

                if physx.is_gpu_enabled():
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()

                if env_idx.numel() == 0:
                    self.subtask_objs[0].set_pose(
                        self.agent.tcp_pose * Pose.create(obj_init_raw_pose_wrt_tcp)
                    )
                    break
                # ---------------------------------------------------

            if self.target_randomization:
                b = len(env_idx)

                xyz = torch.zeros((b, 3))
                xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
                xyz += self.subtask_goals[0].pose.p

                qs = quaternion_raw_multiply(
                    randomization.random_quaternions(
                        b, lock_x=True, lock_y=True, lock_z=False
                    ),
                    self.subtask_objs[0].pose.q,
                )
                self.subtask_objs[0].set_pose(Pose.create_from_pq(xyz, qs))

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj_pos = self.subtask_objs[0].pose.p
            goal_pos = self.subtask_goals[0].pose.p
            rest_pos = self.ee_rest_world_pose.p
            tcp_pos = self.agent.tcp_pose.p

            # NOTE (arth): reward "steps" are as follows:
            #       - reaching_reward
            #       - is_grasped_reward
            #       - if grasped and not at goal
            #           - obj to goal reward
            #       - if at goal
            #           - rest reward
            #       - if at rest
            #           - static reward
            #       - success_reward
            # ---------------------------------------------------
            # CONDITION CHECKERS
            # ---------------------------------------------------

            obj_to_goal_dist = torch.norm(obj_pos - goal_pos, dim=1)
            tcp_to_goal_dist = torch.norm(tcp_pos - goal_pos, dim=1)

            obj_not_at_goal = ~info["obj_at_goal"]
            obj_not_at_goal_reward = torch.zeros_like(reward[obj_not_at_goal])

            obj_at_goal_maybe_dropped = info["obj_at_goal"]
            obj_at_goal_maybe_dropped_reward = torch.zeros_like(
                reward[obj_at_goal_maybe_dropped]
            )

            ee_to_rest_dist = torch.norm(tcp_pos - rest_pos, dim=1)
            ee_rest = obj_at_goal_maybe_dropped & (
                ee_to_rest_dist <= self.place_cfg.ee_rest_thresh
            )
            ee_rest_reward = torch.zeros_like(reward[ee_rest])

            # ---------------------------------------------------

            # penalty for ee jittering too much
            ee_vel = self.agent.tcp.linear_velocity
            ee_still_rew = 1 - torch.tanh(torch.norm(ee_vel, dim=1) / 5)
            reward += ee_still_rew

            # penalty for object moving too much
            obj_vel = torch.norm(
                self.subtask_objs[0].linear_velocity, dim=1
            ) + torch.norm(self.subtask_objs[0].angular_velocity, dim=1)
            obj_still_rew = 3 * (1 - torch.tanh(obj_vel / 5))
            reward += obj_still_rew

            # success reward
            success_rew = 3 * info["success"]
            reward += success_rew

            # encourage arm and torso in "resting" orientation
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )
            arm_resting_orientation_rew = 2 * (1 - torch.tanh(arm_to_resting_diff))
            reward += arm_resting_orientation_rew

            # ---------------------------------------------------------------
            # colliisions
            step_no_col_rew = 3 * (
                1
                - torch.tanh(
                    3
                    * (
                        torch.clamp(
                            self.robot_force_mult * info["robot_force"],
                            min=self.robot_force_penalty_min,
                        )
                        - self.robot_force_penalty_min
                    )
                )
            )
            reward += step_no_col_rew

            # cumulative collision penalty
            cum_col_under_thresh_rew = (
                2
                * (
                    info["robot_cumulative_force"] < self.robot_cumulative_force_limit
                ).float()
            )
            reward += cum_col_under_thresh_rew
            # ---------------------------------------------------------------

            if torch.any(obj_not_at_goal):
                # ee holding object
                obj_not_at_goal_reward += 2 * info["is_grasped"][obj_not_at_goal]

                # ee and tcp close to goal
                place_rew = 5 * (
                    1
                    - (
                        (
                            torch.tanh(obj_to_goal_dist[obj_not_at_goal])
                            + torch.tanh(tcp_to_goal_dist[obj_not_at_goal])
                        )
                        / 2
                    )
                )
                obj_not_at_goal_reward += place_rew

            if torch.any(obj_at_goal_maybe_dropped):
                # add prev step max rew
                obj_at_goal_maybe_dropped_reward += 7

                # rest reward
                rest_rew = 5 * (
                    1 - torch.tanh(3 * ee_to_rest_dist[obj_at_goal_maybe_dropped])
                )
                obj_at_goal_maybe_dropped_reward += rest_rew

                # additional encourage arm and torso in "resting" orientation
                more_arm_resting_orientation_rew = 2 * (
                    1 - torch.tanh(arm_to_resting_diff[obj_at_goal_maybe_dropped])
                )
                obj_at_goal_maybe_dropped_reward += more_arm_resting_orientation_rew

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][obj_at_goal_maybe_dropped]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                obj_at_goal_maybe_dropped_reward += base_still_rew

            if torch.any(ee_rest):
                ee_rest_reward += 2

                qvel = self.agent.robot.qvel[..., :-2][ee_rest]
                static_rew = 1 - torch.tanh(torch.norm(qvel, dim=1))
                ee_rest_reward += static_rew

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][ee_rest]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                ee_rest_reward += base_still_rew

            # add rewards to specific envs
            reward[obj_not_at_goal] += obj_not_at_goal_reward
            reward[obj_at_goal_maybe_dropped] += obj_at_goal_maybe_dropped_reward
            reward[ee_rest] += ee_rest_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 33.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
