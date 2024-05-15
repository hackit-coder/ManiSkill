import json
import yaml
from dacite import from_dict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Tuple, List
import shortuuid

"""
Task Planner Dataclasses
"""


@dataclass
class Subtask:
    type: str = field(init=False)
    uid: str = field(init=False)

    def __post_init__(self):
        assert self.type in ["pick", "place", "navigate"]
        self.uid = self.type + "_" + shortuuid.ShortUUID().random(length=6)


@dataclass
class SubtaskConfig:
    task_id: int
    horizon: int = -1

    def __post_init__(self):
        assert self.horizon > 0


@dataclass
class PickSubtask(Subtask):
    obj_id: str

    def __post_init__(self):
        self.type = "pick"
        super().__post_init__()


@dataclass
class PickSubtaskConfig(SubtaskConfig):
    task_id: int = 0
    horizon: int = 200
    ee_rest_thresh: float = 0.05

    def __post_init__(self):
        assert self.ee_rest_thresh >= 0


@dataclass
class PlaceSubtask(Subtask):
    obj_id: str
    goal_pos: Union[str, Tuple[float, float, float], List[Tuple[float, float, float]]]

    def __post_init__(self):
        self.type = "place"
        super().__post_init__()
        if isinstance(self.goal_pos, str):
            self.goal_pos = [float(coord) for coord in self.goal_pos.split(",")]


@dataclass
class PlaceSubtaskConfig(SubtaskConfig):
    task_id: int = 1
    horizon: int = 200
    ee_rest_thresh: float = 0.05
    obj_goal_thresh: float = 0.15

    def __post_init__(self):
        assert self.obj_goal_thresh >= 0
        assert self.ee_rest_thresh >= 0


@dataclass
class NavigateSubtask(Subtask):
    obj_id: Union[str, None] = None
    goal_pos: Union[
        str, Tuple[float, float, float], List[Tuple[float, float, float]], None
    ] = None

    def __post_init__(self):
        self.type = "navigate"
        super().__post_init__()
        if isinstance(self.goal_pos, str):
            self.goal_pos = [float(coord) for coord in self.goal_pos.split(",")]


@dataclass
class NavigateSubtaskConfig(SubtaskConfig):
    task_id: int = 2
    horizon: int = 200
    ee_rest_thresh: float = 0.05
    navigated_sucessfully_dist: float = 2

    def __post_init__(self):
        assert self.ee_rest_thresh >= 0


@dataclass
class TaskPlan:
    subtasks: List[Subtask]
    build_config_name: Union[str, None] = None
    init_config_name: Union[str, None] = None


"""
Reading Task Plan from file
"""


@dataclass
class PlanData:
    dataset: str
    plans: List[TaskPlan]


def plan_data_from_file(cfg_path: str = None) -> PlanData:
    cfg_path: Path = Path(cfg_path)
    assert cfg_path.exists(), f"Path {cfg_path} not found"

    suffix = Path(cfg_path).suffix
    if suffix == ".json":
        with open(cfg_path, "rb") as f:
            plan_data = json.load(f)
    elif suffix == ".yml":
        with open(cfg_path) as f:
            plan_data = yaml.safe_load(f)
    else:
        print(f"{suffix} not supported")

    plans = []
    for task_plan_data in plan_data["plans"]:
        build_config_name = task_plan_data["build_config_name"]
        init_config_name = task_plan_data["init_config_name"]
        subtasks = []
        for subtask in task_plan_data["subtasks"]:
            subtask_type = subtask["type"]
            if subtask_type == "pick":
                cls = PickSubtask
            elif subtask_type == "place":
                cls = PlaceSubtask
            elif subtask_type == "navigate":
                cls = NavigateSubtask
            else:
                raise NotImplementedError(f"Subtask {subtask_type} not implemented yet")
            subtasks.append(from_dict(data_class=cls, data=subtask))
        plans.append(
            TaskPlan(
                subtasks=subtasks,
                build_config_name=build_config_name,
                init_config_name=init_config_name,
            )
        )

    return PlanData(dataset=plan_data["dataset"], plans=plans)
