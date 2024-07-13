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
        goal_type="cylinder",
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
            super()._initialize_episode(env_idx, options)

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

            # penalty for object moving too much when not grasped
            obj_vel = torch.norm(
                self.subtask_objs[0].linear_velocity, dim=1
            ) + torch.norm(self.subtask_objs[0].angular_velocity, dim=1)
            obj_vel[info["is_grasped"]] = 0
            obj_still_rew = 3 * (1 - torch.tanh(obj_vel / 5))
            reward += obj_still_rew

            # success reward
            success_rew = 6 * info["success"]
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
        max_reward = 36.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
