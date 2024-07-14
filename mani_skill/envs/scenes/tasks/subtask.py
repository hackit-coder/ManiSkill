from typing import List, Dict
from pathlib import Path
from collections import defaultdict

import torch
import sapien.physx as physx

from mani_skill.utils.structs import Pose
from .sequential_task import SequentialTaskEnv, GOAL_POSE_Q
from .planner import TaskPlan, Subtask


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
        # additional spawn randomization, shouldn't need to change
        spawn_data_fp=None,
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

        # spawn vals
        self.spawn_data_fp = Path(spawn_data_fp)
        assert self.spawn_data_fp.exists()

        # force reward hparams
        self.robot_force_mult = robot_force_mult
        self.robot_force_penalty_min = robot_force_penalty_min
        self.robot_cumulative_force_limit = robot_cumulative_force_limit

        # additional target obj randomization
        self.target_randomization = target_randomization

        self.subtask_cfg = getattr(self, "subtask_cfg", None)
        assert (
            self.subtask_cfg is not None
        ), "Need to designate self.subtask_config (in extending env)"

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # RECONFIGURE AND INIT
    # -------------------------------------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
        return super().reset(*args, **kwargs)

    def _after_reconfigure(self, options):
        force_rew_ignore_links = [
            self.agent.finger1_link,
            self.agent.finger2_link,
        ]
        self.force_articulation_link_ids = [
            link.name
            for link in self.agent.robot.get_links()
            if link not in force_rew_ignore_links
        ]
        self.spawn_data = torch.load(self.spawn_data_fp, map_location=self.device)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            options["sample_place_goal_pos"] = False
            super()._initialize_episode(env_idx, options)

            current_subtask = self.task_plan[0]
            batched_spawn_data = defaultdict(list)
            for subtask_uid in current_subtask.composite_subtask_uids:
                spawn_data: Dict[str, torch.Tensor] = self.spawn_data[subtask_uid]
                spawn_selection_idx = None
                for k, v in spawn_data.items():
                    if spawn_selection_idx is None:
                        spawn_selection_idx = torch.randint(
                            low=0, high=len(v), size=(1,)
                        )
                    batched_spawn_data[k].append(v[spawn_selection_idx])
            batched_spawn_data = dict(
                (k, torch.cat(v, dim=0)) for k, v in batched_spawn_data.items()
            )
            if "robot_pos" in batched_spawn_data:
                self.agent.robot.set_pose(
                    Pose.create_from_pq(p=batched_spawn_data["robot_pos"])
                )
            if "robot_qpos" in batched_spawn_data:
                self.agent.robot.set_qpos(batched_spawn_data["robot_qpos"])
            if "obj_raw_pose_wrt_tcp" in batched_spawn_data:
                if physx.is_gpu_enabled():
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()
                self.subtask_objs[0].set_pose(
                    self.agent.tcp_pose
                    * Pose.create(batched_spawn_data["obj_raw_pose_wrt_tcp"])
                )
            if "goal_pos" in batched_spawn_data:
                self.subtask_goals[0].set_pose(
                    Pose.create_from_pq(q=GOAL_POSE_Q, p=batched_spawn_data["goal_pos"])
                )
                self.task_plan[0].goal_pos = batched_spawn_data["goal_pos"]
                self.task_plan[0].goal_rectangle_corners = batched_spawn_data[
                    "goal_rectangle_corners"
                ]

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
