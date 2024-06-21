from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import sapien
import torch
import torch.random
import numpy as np

from mani_skill.agents.robots import Fetch
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.pose import vectorize_pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.visualization.misc import observations_to_images, tile_images

from .planner import (
    PickSubtask,
    PickSubtaskConfig,
    PlaceSubtask,
    PlaceSubtaskConfig,
    NavigateSubtask,
    NavigateSubtaskConfig,
    Subtask,
    SubtaskConfig,
    TaskPlan,
)

UNIQUE_SUCCESS_SUBTASK_TYPE = 100_000
HIDDEN_POSITION = [99999, 99999, 99999]


def all_equal(array: list):
    return len(set(array)) == 1


def all_same_type(array: list):
    return len(set(map(type, array))) == 1


@register_env("SequentialTask-v0")
class SequentialTaskEnv(SceneManipulationEnv):
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

    SUPPORTED_ROBOTS = ["fetch"]
    agent: Fetch

    # TODO (arth): add open/close articulation tasks
    EE_REST_POS_WRT_BASE = (0.5, 0, 1.25)
    pick_cfg = PickSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
    )
    place_cfg = PlaceSubtaskConfig(
        horizon=200,
        obj_goal_thresh=0.15,
        ee_rest_thresh=0.05,
    )
    navigate_cfg = NavigateSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        navigated_sucessfully_dist=2,
    )
    task_cfgs: Dict[str, SubtaskConfig] = dict(
        pick=pick_cfg,
        place=place_cfg,
        navigate=navigate_cfg,
    )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=50,
            gpu_memory_cfg=GPUMemoryConfig(
                temp_buffer_capacity=2**24,
                max_rigid_contact_count=2**23,
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
            ),
        )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        require_build_configs_repeated_equally_across_envs=True,
        **kwargs,
    ):
        assert all_equal(
            [len(plan.subtasks) for plan in task_plans]
        ), "All parallel task plans must be the same length"
        assert all(
            [
                all_same_type(parallel_subtasks)
                for parallel_subtasks in zip(*[plan.subtasks for plan in task_plans])
            ]
        ), "All parallel task plans must have same subtask types in same order"

        self._require_build_configs_repeated_equally_across_envs = (
            require_build_configs_repeated_equally_across_envs
        )

        self.base_task_plans: Dict[str, List[TaskPlan]] = defaultdict(list)
        for tp in task_plans:
            self.base_task_plans[tp.build_config_name].append(tp)

        self.seq_task_len = len(task_plans[0].subtasks)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # PROCESS TASKS
    # -------------------------------------------------------------------------------------------------

    def process_task_plan(self, sampled_subtask_lists: List[List[Subtask]]):

        self.subtask_objs: List[Actor] = []
        self.subtask_goals: List[Actor] = []

        # build new merged task_plan and merge actors of parallel task plants
        self.task_plan: List[Subtask] = []
        last_subtask0 = None
        for subtask_num, parallel_subtasks in enumerate(zip(*sampled_subtask_lists)):
            subtask0: Subtask = parallel_subtasks[0]

            if isinstance(subtask0, PickSubtask):
                parallel_subtasks: List[PickSubtask]
                merged_obj_name = f"obj_{subtask_num}"
                self.subtask_objs.append(
                    self._create_merged_actor_from_subtasks(
                        parallel_subtasks, name=merged_obj_name
                    )
                )

                self.premade_place_goal_list[subtask_num].set_pose(
                    sapien.Pose(p=HIDDEN_POSITION)
                )
                self.subtask_goals.append(None)

                self.task_plan.append(PickSubtask(obj_id=merged_obj_name))

            elif isinstance(subtask0, PlaceSubtask):
                parallel_subtasks: List[PlaceSubtask]
                merged_obj_name = f"obj_{subtask_num}"
                self.subtask_objs.append(
                    self._create_merged_actor_from_subtasks(
                        parallel_subtasks, name=merged_obj_name
                    )
                )

                Bs, BCs, BAs = [], [], []
                for subtask in parallel_subtasks:
                    grcs = np.array(
                        subtask.goal_rectangle_corners
                        if subtask.goal_rectangle_probs is None
                        else self.np_random.choice(
                            subtask.goal_rectangle_corners,
                            p=subtask.goal_rectangle_probs,
                        )
                    )
                    grcs[..., 2] += self.place_cfg.obj_goal_thresh * 2 / 3
                    Bs.append(grcs[1])
                    BCs.append(grcs[2] - grcs[1])
                    BAs.append(grcs[0] - grcs[1])
                Bs, BCs, BAs = np.array(Bs), np.array(BCs), np.array(BAs)

                u, v = self.np_random.uniform(
                    low=0, high=1, size=(len(parallel_subtasks), 1)
                ), self.np_random.uniform(
                    low=0, high=1, size=(len(parallel_subtasks), 1)
                )

                goal_pos = ((BCs * u + BAs * v) + Bs).tolist()
                self.premade_place_goal_list[subtask_num].set_pose(
                    Pose.create_from_pq(p=torch.tensor(goal_pos))
                )
                self.subtask_goals.append(self.premade_place_goal_list[subtask_num])

                self.task_plan.append(
                    PlaceSubtask(
                        obj_id=merged_obj_name,
                        goal_pos=goal_pos,
                    )
                )

            elif isinstance(subtask0, NavigateSubtask):
                self.premade_place_goal_list[subtask_num].set_pose(
                    sapien.Pose(p=HIDDEN_POSITION)
                )
                self.subtask_goals.append(None)

                if isinstance(last_subtask0, PickSubtask):
                    last_subtask_obj = self.subtask_objs[-1]
                    self.subtask_objs.append(last_subtask_obj)
                    self.task_plan.append(NavigateSubtask(obj_id=last_subtask_obj.name))
                else:
                    self.subtask_objs.append(None)
                    self.task_plan.append(NavigateSubtask())

            else:
                raise AttributeError(
                    f"{subtask0.type} {type(subtask0)} not yet supported"
                )

            last_subtask0 = subtask0

        # add navigation goals for each Navigate Subtask depending on following subtask
        last_subtask = None
        for i, (subtask_obj, subtask_goal, subtask) in enumerate(
            zip(self.subtask_objs, self.subtask_goals, self.task_plan)
        ):
            if isinstance(last_subtask, NavigateSubtask):
                if isinstance(subtask, PickSubtask):
                    self.subtask_goals[i - 1] = subtask_obj
                elif isinstance(subtask, PlaceSubtask):
                    self.subtask_goals[i - 1] = subtask_goal
            last_subtask = subtask

        self.task_horizons = torch.tensor(
            [self.task_cfgs[subtask.type].horizon for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )
        self.task_ids = torch.tensor(
            [self.task_cfgs[subtask.type].task_id for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )

        # TODO (arth): figure out how to change horizon after task inited
        # self.max_episode_steps = torch.sum(self.task_horizons)

    def _get_actor_entity(self, actor_id: str, env_num: int):
        actor = self.scene_builder.movable_objects[actor_id]
        return actor._objs[actor._scene_idxs.tolist().index(env_num)]

    def _create_merged_actor_from_subtasks(
        self,
        parallel_subtasks: Union[List[PickSubtask], List[PlaceSubtask]],
        name: str = None,
    ):
        merged_obj = Actor.create_from_entities(
            [
                self._get_actor_entity(actor_id=f"env-{i}_{subtask.obj_id}", env_num=i)
                for i, subtask in enumerate(parallel_subtasks)
            ],
            scene=self.scene,
            scene_idxs=torch.arange(self.num_envs, dtype=int),
        )
        if name is not None:
            merged_obj.name = name
        return merged_obj

    def _make_goal(
        self,
        pos: Union[Tuple[float, float, float], List[Tuple[float, float, float]]] = None,
        radius=0.15,
        name="goal_site",
    ):
        goal = actors.build_sphere(
            self.scene,
            radius=radius,
            color=[0, 1, 0, 1],
            name=name,
            body_type="kinematic",
            add_collision=False,
        )
        if pos is not None:
            if len(pos) == self.num_envs:
                goal.set_pose(Pose.create_from_pq(p=pos))
            else:
                goal.set_pose(sapien.Pose(p=pos))
        self._hidden_objects.append(goal)
        return goal

    @property
    def ee_rest_world_pose(self) -> Pose:
        return self.agent.base_link.pose * self.ee_rest_pos_wrt_base

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # RESET/RECONFIGURE HANDLING
    # -------------------------------------------------------------------------------------------------

    def _load_scene(self, options):
        self.premade_place_goal_list: List[Actor] = [
            self._make_goal(
                radius=self.place_cfg.obj_goal_thresh,
                name=f"goal_{subtask_num}",
            )
            for subtask_num in range(self.seq_task_len)
        ]

        self.build_config_idx_to_task_plans: Dict[int, List[TaskPlan]] = dict()
        for bc in self.base_task_plans.keys():
            self.build_config_idx_to_task_plans[
                self.scene_builder.build_config_names_to_idxs[bc]
            ] = self.base_task_plans[bc]

        num_bcis = len(self.build_config_idx_to_task_plans.keys())

        assert (
            not self._require_build_configs_repeated_equally_across_envs
            or self.num_envs % num_bcis == 0
        ), f"These task plans cover {num_bcis} build configs, but received {self.num_envs} envs. Either change the task plan list, change num_envs, or set require_build_configs_repeated_equally_across_envs=False. Note if require_build_configs_repeated_equally_across_envs=False and num_envs % num_build_configs != 0, then a) if num_envs > num_build_configs, then some build configs might be built in more parallel envs than others (meaning associated task plans will be sampled more frequently), and b) if num_envs < num_build_configs, then some build configs might not be built at all (meaning associated task plans will not be used)."

        # if num_bcis < self.num_envs, repeat bcis and truncate at self.num_envs
        # TODO (arth): decide if this is a good option
        #   (e.g. if user wants 1 build config / env but accidentally passed num_envs
        #   value that was 1 too large, then they wouldn't know)
        self.build_config_idxs = np.repeat(
            sorted(list(self.build_config_idx_to_task_plans.keys())),
            np.ceil(self.num_envs / num_bcis),
        )[: self.num_envs].tolist()
        self.num_task_plans_per_bci = torch.tensor(
            [
                len(self.build_config_idx_to_task_plans[bci])
                for bci in self.build_config_idxs
            ],
            device=self.device,
        )
        super()._load_scene(options)
        self.ee_rest_pos_wrt_base = Pose.create_from_pq(
            p=self.EE_REST_POS_WRT_BASE, device=self.device
        )
        self.subtask_pointer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.subtask_steps_left = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.ee_rest_goal = self._make_goal(
            radius=0.05,
            name="ee_rest_goal",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options):
        with torch.device(self.device):
            self.task_plan_idxs = options.pop("task_plan_idxs", None)
            if self.task_plan_idxs is None:
                low = torch.zeros(len(self.build_config_idxs))
                high = self.num_task_plans_per_bci
                size = (len(self.build_config_idxs),)
                self.task_plan_idxs: List[int] = (
                    (torch.randint(2**63 - 1, size=size) % (high - low) + low)
                    .int()
                    .tolist()
                )
            sampled_task_plans = [
                self.build_config_idx_to_task_plans[bci][tpi]
                for bci, tpi in zip(self.build_config_idxs, self.task_plan_idxs)
            ]
            self.init_config_idxs = [
                self.scene_builder.init_config_names_to_idxs[tp.init_config_name]
                for tp in sampled_task_plans
            ]
            super()._initialize_episode(env_idx, options)
            self.process_task_plan(
                sampled_subtask_lists=[tp.subtasks for tp in sampled_task_plans]
            )

            self.subtask_pointer[env_idx] = 0
            self.subtask_steps_left[env_idx] = self.task_cfgs[
                self.task_plan[0].type
            ].horizon

            self.resting_qpos = torch.tensor(self.agent.keyframes["rest"].qpos[3:-2])

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # STATE RESET
    # -------------------------------------------------------------------------------------------------

    def get_state_dict(self):
        state_dict = super().get_state_dict()

        state_dict["task_plan_idxs"] = self.task_plan_idxs
        state_dict["build_config_idxs"] = self.build_config_idxs
        state_dict["init_config_idxs"] = self.init_config_idxs

        state_dict["subtask_pointer"] = self.subtask_pointer
        state_dict["subtask_steps_left"] = self.subtask_steps_left

        return state_dict

    def set_state_dict(self, state_dict: Dict):

        task_plan_idxs = state_dict.pop("task_plan_idxs")
        build_config_idxs = state_dict.pop("build_config_idxs")
        init_config_idxs = state_dict.pop("init_config_idxs")

        assert torch.all(
            torch.tensor(self.build_config_idxs) == torch.tensor(build_config_idxs)
        ), f"Please pass the same task plan list when creating this env as was used in this state dict; currently built build_config_idxs={self.build_config_idxs}, state dict build_config_idxs={build_config_idxs}"

        self._initialize_episode(
            torch.arange(self.num_envs), options=dict(task_plan_idxs=task_plan_idxs)
        )

        assert torch.all(
            torch.tensor(self.init_config_idxs) == torch.tensor(init_config_idxs)
        ), f"Please pass the same task plan list when creating this env as was used in this state dict; currently init'd init_config_idxs={self.init_config_idxs}, state dict init_config_idxs={init_config_idxs}"

        self.subtask_pointer = state_dict.pop("subtask_pointer")
        self.subtask_steps_left = state_dict.pop("subtask_steps_left")

        super().set_state_dict(state_dict)

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # SUBTASK STATUS CHECKERS/UPDATERS
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        subtask_success, success_checkers = self._subtask_check_success()

        self.subtask_pointer[subtask_success] += 1
        success = self.subtask_pointer >= len(self.task_plan)

        self.subtask_steps_left -= 1
        update_subtask_horizon = subtask_success & ~success
        self.subtask_steps_left[update_subtask_horizon] = self.task_horizons[
            self.subtask_pointer[update_subtask_horizon]
        ]

        fail = (self.subtask_steps_left <= 0) & ~success

        subtask_type = torch.full_like(
            self.subtask_pointer, UNIQUE_SUCCESS_SUBTASK_TYPE
        )
        subtask_type[~success] = self.task_ids[self.subtask_pointer[[~success]]]

        return dict(
            success=success,
            fail=fail,
            subtask=self.subtask_pointer,
            subtask_type=subtask_type,
            subtasks_steps_left=self.subtask_steps_left,
            **success_checkers,
        )

    def _subtask_check_success(self):
        subtask_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        success_checkers = dict()

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            subtask: Subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            if isinstance(subtask, PickSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._pick_check_success(
                    self.subtask_objs[subtask_num],
                    env_idx,
                    ee_rest_thresh=self.pick_cfg.ee_rest_thresh,
                )
            elif isinstance(subtask, PlaceSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._place_check_success(
                    self.subtask_objs[subtask_num],
                    self.subtask_goals[subtask_num],
                    env_idx,
                    obj_goal_thresh=self.place_cfg.obj_goal_thresh,
                    ee_rest_thresh=self.place_cfg.ee_rest_thresh,
                )
            elif isinstance(subtask, NavigateSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._navigate_check_success(
                    self.subtask_objs[subtask_num],
                    self.subtask_goals[subtask_num],
                    env_idx,
                    ee_rest_thresh=self.place_cfg.ee_rest_thresh,
                )
            else:
                raise AttributeError(f"{subtask.type} {type(subtask)} not supported")

            for k, v in subtask_success_checkers.items():
                if k not in success_checkers:
                    success_checkers[k] = torch.zeros(
                        self.num_envs, device=self.device, dtype=torch.bool
                    )
                success_checkers[k][env_idx] = subtask_success_checkers[k]

        return subtask_success, success_checkers

    def _pick_check_success(
        self,
        obj: Actor,
        env_idx: torch.Tensor,
        ee_rest_thresh: float = 0.05,
    ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 3:-2] - self.resting_qpos
        )
        robot_rest = torch.all(
            robot_rest_dist < self.pick_cfg.robot_resting_qpos_tolerance_grasping, dim=1
        )
        is_static = self.agent.is_static(threshold=0.2)[env_idx]
        return (
            is_grasped & ee_rest & robot_rest & is_static,
            dict(
                is_grasped=is_grasped,
                ee_rest=ee_rest,
                robot_rest=robot_rest,
                is_static=is_static,
            ),
        )

    def _place_check_success(
        self,
        obj: Actor,
        obj_goal: Actor,
        env_idx: torch.Tensor,
        obj_goal_thresh: float = 0.15,
        ee_rest_thresh: float = 0.05,
    ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        obj_at_goal = (
            torch.norm(obj.pose.p[env_idx] - obj_goal.pose.p[env_idx], dim=1)
            <= obj_goal_thresh
        )
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 3:-2] - self.resting_qpos
        )
        robot_rest = torch.all(
            robot_rest_dist < self.pick_cfg.robot_resting_qpos_tolerance, dim=1
        )
        is_static = self.agent.is_static(threshold=0.2)[env_idx]
        return (
            ~is_grasped & obj_at_goal & ee_rest & robot_rest & is_static,
            dict(
                is_grasped=is_grasped,
                obj_at_goal=obj_at_goal,
                ee_rest=ee_rest,
                robot_rest=robot_rest,
                is_static=is_static,
            ),
        )

    def _navigate_check_success(
        self,
        obj: Union[Actor, None],
        goal: Actor,
        env_idx: torch.Tensor,
        ee_rest_thresh: float = 0.05,
    ):
        if obj is not None:
            is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        else:
            is_grasped = torch.zeros_like(env_idx, dtype=torch.bool)

        goal_pose_wrt_base = self.agent.base_link.pose.inv() * goal.pose
        targ = goal_pose_wrt_base.p[..., :2][env_idx]
        uc_targ = targ / torch.norm(targ, dim=1).unsqueeze(-1).expand(*targ.shape)
        rots = torch.sign(uc_targ[..., 1]) * torch.arccos(uc_targ[..., 0])
        oriented_correctly = torch.abs(rots) < 0.5

        navigated_close = (
            torch.norm(targ, dim=1) <= self.navigate_cfg.navigated_sucessfully_dist
        )
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 3:-2] - self.resting_qpos
        )
        rest_tolerance = (
            self.pick_cfg.robot_resting_qpos_tolerance
            if obj is None
            else self.pick_cfg.robot_resting_qpos_tolerance_grasping
        )
        robot_rest = torch.all(robot_rest_dist < rest_tolerance, dim=1)
        is_static = self.agent.is_static(threshold=0.2)[env_idx]
        navigate_success = (
            oriented_correctly & navigated_close & ee_rest & robot_rest & is_static
        )
        if obj is not None:
            navigate_success &= is_grasped
        return (
            navigate_success,
            dict(
                is_grasped=is_grasped,
                oriented_correctly=oriented_correctly,
                navigated_close=navigated_close,
                ee_rest=ee_rest,
                robot_rest=robot_rest,
                is_static=is_static,
            ),
        )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------

    def _get_obs_agent(self):
        agent_state = super()._get_obs_agent()
        agent_state["qpos"][..., :3] = 0
        return agent_state

    # NOTE (arth): for now, define keys that will always be added to obs. leave it to
    #       wrappers or task-specific envs to mask out unnecessary vals
    #       - subtasks that don't need that obs will set some default value
    #       - subtasks which need that obs will set value depending on subtask params
    def _get_obs_extra(self, info: Dict):
        base_pose_inv = self.agent.base_link.pose.inv()

        # all subtasks will have same computation for
        #       - tcp_pose_wrt_base :   tcp always there and is same link
        tcp_pose_wrt_base = vectorize_pose(base_pose_inv * self.agent.tcp.pose)

        #       - obj_pose_wrt_base :   different objs per subtask (or no obj)
        #       - goal_pos_wrt_base :   different goals per subtask (or no goal)
        obj_pose_wrt_base = torch.zeros(
            self.num_envs, 7, device=self.device, dtype=torch.float
        )
        goal_pos_wrt_base = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            if self.subtask_objs[subtask_num] is not None:
                obj_pose_wrt_base[env_idx] = vectorize_pose(
                    base_pose_inv * self.subtask_objs[subtask_num].pose
                )[env_idx]
            if self.subtask_goals[subtask_num] is not None:
                goal_pos_wrt_base[env_idx] = (
                    base_pose_inv * self.subtask_goals[subtask_num].pose
                ).p[env_idx]

        # already computed during evaluation is
        #       - is_grasped    :   part of success criteria (or set default)
        is_grasped = info["is_grasped"]

        return dict(
            tcp_pose_wrt_base=tcp_pose_wrt_base,
            obj_pose_wrt_base=obj_pose_wrt_base,
            goal_pos_wrt_base=goal_pos_wrt_base,
            is_grasped=is_grasped,
        )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD (Ignored here)
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): this env does not have dense rewards since rewards are used for training subtasks.
    #       If need to train a subtask, extend this class to define a subtask
    # -------------------------------------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.subtask_pointer

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # CAMERAS, SENSORS, AND RENDERING
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): also included the old "cameras" mode from MS2 since HAB renders this way
    # -------------------------------------------------------------------------------------------------

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        # room_camera_pose = sapien_utils.look_at(
        #     [2, -0.8, 1.75], [-1.9, -1, 0]
        # )  # fov 1.75
        # room_camera_pose = sapien_utils.look_at(
        #     [-0.3, 0, 2], [1.5, -4.46, 1]
        # )  # fov 1.5
        # room_camera_pose = sapien_utils.look_at(
        #     [1.5, -2.4, 2], [0.5, -3.7, 1.5]
        # )  # fov 1.75
        # room_camera_pose = sapien_utils.look_at(
        #     [-1, -0.5, 3], [0.4, -5.4, 0.4]
        # )  # fov 1.3
        # room_camera_pose = sapien_utils.look_at(
        #     [-1.5, -2.5, 3], [-0.6, -1.8, 0]
        # )  # fov 1.75
        # room_camera_pose = sapien_utils.look_at([3.7, 1, 3], [0, -3, 0])  # fov 1.75
        # room_camera_config = CameraConfig(
        #     "render_camera",
        #     room_camera_pose,
        #     512,
        #     512,
        #     1.75,
        #     0.01,
        #     10,
        # )

        # return room_camera_config

        # this camera follows the robot around (though might be in walls if the space is cramped)
        robot_camera_pose = sapien_utils.look_at([-0.2, 0.5, 1], ([0.2, -0.2, 0]))
        robot_camera_config = CameraConfig(
            "render_camera",
            robot_camera_pose,
            512,
            512,
            1.75,
            0.01,
            10,
            mount=self.agent.torso_lift_link,
        )
        return robot_camera_config

    def render_cameras(self):
        for obj in self._hidden_objects:
            obj.hide_visual()
        images = []
        self.scene.update_render()
        self.capture_sensor_data()
        sensor_images = self.get_sensor_obs()
        for sensor_images in sensor_images.values():
            images.extend(observations_to_images(sensor_images))
        return tile_images([self.render_rgb_array()] + images)

    def render_rgb_array(self):
        self.ee_rest_goal.set_pose(self.ee_rest_world_pose)
        return super().render_rgb_array()

    def render_human(self):
        self.ee_rest_goal.set_pose(self.ee_rest_world_pose)
        return super().render_human()

    def render(self):
        if self.render_mode == "cameras":
            return self.render_cameras()

        return super().render()

    # -------------------------------------------------------------------------------------------------
