from .sequential_task import SequentialTaskEnv
from .planner import TaskPlan, Subtask

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
import sapien.physx as physx

import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Any, Dict, List


@register_env("SubtaskTrain-v0", max_episode_steps=200)
class SubtaskTrainEnv(SequentialTaskEnv):
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

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        # spawn randomization
        randomize_arm=False,
        randomize_base=False,
        randomize_loc=False,
        # additional spawn randomization, shouldn't need to change
        spawn_loc_radius=2,
        # colliison tracking
        robot_force_mult=0,
        robot_force_penalty_min=0,
        robot_cumulative_force_limit=torch.inf,
        # additional randomization
        target_randomization=False,
        **kwargs,
    ):
        tp0 = task_plans[0]
        assert len(tp0.subtasks) == 1 and isinstance(
            tp0.subtasks[0], Subtask
        ), f"Task plans for {self.__class__.__name__} must be one {Subtask.__name__} long"

        # randomization vals
        self.randomize_arm = randomize_arm
        self.randomize_base = randomize_base
        self.randomize_loc = randomize_loc
        self.spawn_loc_radius = spawn_loc_radius

        # force reward hparams
        self.robot_force_mult = robot_force_mult
        self.robot_force_penalty_min = robot_force_penalty_min
        self.robot_cumulative_force_limit = robot_cumulative_force_limit

        # additional target obj randomization
        self.target_randomization = target_randomization

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # COLLISION TRACKING
    # -------------------------------------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
        return super().reset(*args, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------

    def _after_reconfigure(self, options):
        force_rew_ignore_links = [
            self.agent.finger1_link,
            self.agent.finger2_link,
            self.agent.tcp,
        ]
        self.force_articulation_link_ids = [
            link.name
            for link in self.agent.robot.get_links()
            if link not in force_rew_ignore_links
        ]

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            original_env_idx = env_idx.clone()

            super()._initialize_episode(env_idx, options)

            robot_init_p, robot_init_q, robot_init_qpos = (
                self.agent.robot.pose.p.clone(),
                self.agent.robot.pose.q.clone(),
                self.agent.robot.qpos.clone(),
            )
            # keep going until no collisions
            while True:

                centers = self.subtask_objs[0].pose.p[env_idx, :2]
                navigable_positions = []
                for env_num, center in zip(env_idx, centers):
                    positions = torch.tensor(
                        self.scene_builder.navigable_positions[env_num]
                    )
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

                self.resting_qpos = torch.tensor(
                    self.agent.keyframes["rest"].qpos[3:-2]
                )

                # NOTE (arth): it is assumed that scene builder spawns agent with some qpos
                qpos = self.agent.robot.get_qpos()

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
                    qpos[env_idx, 5:6] += torch.clamp(
                        torch.normal(0, 0.05, qpos[env_idx, 5:6].shape), -0.1, 0.1
                    ).to(self.device)
                    qpos[env_idx, 7:-2] += torch.clamp(
                        torch.normal(0, 0.05, qpos[env_idx, 7:-2].shape), -0.1, 0.1
                    ).to(self.device)
                self.agent.reset(qpos)

                robot_init_p[env_idx] = self.agent.robot.pose.p[env_idx].clone()
                robot_init_q[env_idx] = self.agent.robot.pose.q[env_idx].clone()
                robot_init_qpos[env_idx] = self.agent.robot.qpos[env_idx].clone()

                if physx.is_gpu_enabled():
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()
                self.scene.step()

                robot_force = (
                    self.agent.robot.get_net_contact_forces(
                        self.force_articulation_link_ids
                    )
                    .norm(dim=-1)
                    .sum(dim=-1)
                )

                self.scene_builder.initialize(original_env_idx, self.init_config_idxs)
                self.agent.reset(robot_init_qpos)
                self.agent.robot.set_pose(
                    Pose.create_from_pq(p=robot_init_p, q=robot_init_q)
                )

                if torch.all((robot_force < 1e-3)[env_idx]):
                    break

                env_idx = torch.where(robot_force >= 1e-3)[0]

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        with torch.device(self.device):
            infos = super().evaluate()

            # set to zero in case we use continuous task
            #   this way, if the termination signal is ignored, env will
            #   still reevaluate success each step
            self.subtask_pointer = torch.zeros_like(self.subtask_pointer)

            robot_force = (
                self.agent.robot.get_net_contact_forces(
                    self.force_articulation_link_ids
                )
                .norm(dim=-1)
                .sum(dim=-1)
            )
            self.robot_cumulative_force += robot_force

            infos.update(
                robot_force=robot_force,
                robot_cumulative_force=self.robot_cumulative_force,
            )

            return infos

    # -------------------------------------------------------------------------------------------------
