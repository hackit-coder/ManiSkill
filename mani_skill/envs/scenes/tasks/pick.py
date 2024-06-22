from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
import sapien.physx as physx
import trimesh

from mani_skill.envs.utils import randomization
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from .planner import (
    TaskPlan,
    PickSubtask,
    PickSubtaskConfig,
)
from .subtask import SubtaskTrainEnv


@register_env("PickSubtaskTrain-v0", max_episode_steps=200)
class PickSubtaskTrainEnv(SubtaskTrainEnv):
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

    pick_cfg = PickSubtaskConfig(
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
            tp0.subtasks[0], PickSubtask
        ), f"Task plans for {self.__class__.__name__} must be one {PickSubtask.__name__} long"

        self.subtask_cfg = self.pick_cfg

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT ROBOT SPAWN RANDOMIZATION
    # -------------------------------------------------------------------------------------------------
    # TODO (arth): maybe check that obj won't fall when noise is added
    # -------------------------------------------------------------------------------------------------

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

                env_idx = torch.where(robot_force >= 1e-3)[0]

                self.scene_builder.initialize(original_env_idx, self.init_config_idxs)
                self.agent.reset(robot_init_qpos)
                self.agent.robot.set_pose(
                    Pose.create_from_pq(p=robot_init_p, q=robot_init_q)
                )

                if env_idx.numel() == 0:
                    break

            if self.target_randomization:
                b = len(env_idx)

                xyz = torch.zeros((b, 3))
                xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
                xyz += self.subtask_objs[0].pose.p
                xyz[..., 2] += 0.005

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
            rest_pos = self.ee_rest_world_pose.p
            tcp_pos = self.agent.tcp_pose.p

            # NOTE (arth): reward "steps" are as follows:
            #       - reaching_reward
            #       - if not grasped
            #           - not_grasped_reward
            #       - is_grasped_reward
            #       - if grasped
            #           - grasped_rewards
            #       - if grasped and ee_at_rest
            #           - static_reward
            #       - success_reward
            # ---------------------------------------------------
            # CONDITION CHECKERS
            # ---------------------------------------------------

            not_grasped = ~info["is_grasped"]
            not_grasped_reward = torch.zeros_like(reward[not_grasped])

            is_grasped = info["is_grasped"]
            is_grasped_reward = torch.zeros_like(reward[is_grasped])

            robot_ee_rest_and_grasped = (
                is_grasped & info["ee_rest"] & info["robot_rest"]
            )
            robot_ee_rest_and_grasped_reward = torch.zeros_like(
                reward[robot_ee_rest_and_grasped]
            )

            # ---------------------------------------------------

            # reaching reward
            tcp_to_obj_dist = torch.norm(obj_pos - tcp_pos, dim=1)
            reaching_rew = 3 * (1 - torch.tanh(5 * tcp_to_obj_dist))
            reward += reaching_rew

            # penalty for ee moving too much when not grasping
            ee_vel = self.agent.tcp.linear_velocity
            ee_still_rew = 1 - torch.tanh(torch.norm(ee_vel, dim=1) / 5)
            reward += ee_still_rew

            # pick reward
            grasp_rew = 2 * info["is_grasped"]
            reward += grasp_rew

            # success reward
            success_rew = 3 * info["success"]
            reward += success_rew

            # encourage arm and torso in "resting" orientation
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )
            arm_resting_orientation_rew = 1 - torch.tanh(arm_to_resting_diff / 5)
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

            if torch.any(not_grasped):
                # penalty for torso moving up and down too much
                tqvel_z = self.agent.robot.qvel[..., 3][not_grasped]
                torso_not_moving_rew = 1 - torch.tanh(5 * torch.abs(tqvel_z))
                not_grasped_reward += torso_not_moving_rew

                # penalty for ee not over obj
                ee_over_obj_rew = 1 - torch.tanh(
                    5
                    * torch.norm(
                        obj_pos[..., :2][not_grasped] - tcp_pos[..., :2][not_grasped],
                        dim=1,
                    )
                )
                not_grasped_reward += ee_over_obj_rew

            if torch.any(is_grasped):
                # not_grasped reward has max of +2
                # so, we add +2 to grasped reward so reward only increases as task proceeds
                is_grasped_reward += 2

                # place reward
                ee_to_rest_dist = torch.norm(
                    tcp_pos[is_grasped] - rest_pos[is_grasped], dim=1
                )
                place_rew = 5 * (1 - torch.tanh(3 * ee_to_rest_dist))
                is_grasped_reward += place_rew

                # arm_to_resting_diff_again
                arm_to_resting_diff_again = torch.norm(
                    self.agent.robot.qpos[is_grasped, 3:-2] - self.resting_qpos,
                    dim=1,
                )
                arm_to_resting_diff_again_reward = 1 - torch.tanh(
                    arm_to_resting_diff_again / 5
                )
                is_grasped_reward += arm_to_resting_diff_again_reward

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][is_grasped]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                is_grasped_reward += base_still_rew

                if torch.any(robot_ee_rest_and_grasped):
                    # increment to encourage robot and ee staying in rest
                    robot_ee_rest_and_grasped_reward += 2

                    qvel = self.agent.robot.qvel[..., :-2][robot_ee_rest_and_grasped]
                    static_rew = 1 - torch.tanh(torch.norm(qvel, dim=1))
                    robot_ee_rest_and_grasped_reward += static_rew

            # add rewards to specific envs
            reward[not_grasped] += not_grasped_reward
            reward[is_grasped] += is_grasped_reward
            reward[robot_ee_rest_and_grasped] += robot_ee_rest_and_grasped_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 27.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
