import pathlib
from pydantic import BaseModel
import yaml
from typing import Callable, TypeVar, Type

T = TypeVar("T", bound="Config")


class Config(BaseModel, extra="forbid"):
    @classmethod
    def from_yaml(cls: Type[T], config_path: pathlib.Path) -> T:
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)
    
class TrainConfig(Config):
    n_instances: int
    batch_size: int
    steps: int
    log_freq: int
    lr: float
    lr_scale: Callable[[int, int], float]