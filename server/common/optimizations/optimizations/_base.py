from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ProgressResult:
    num_train_points_processed: int
    num_errors: int


class Arm(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.promising = True
        self.name = ""

    @property
    @abstractmethod
    def initial_error(self) -> ProgressResult:
        pass

    @abstractmethod
    def can_progress(self) -> bool:
        pass

    @abstractmethod
    def progress(self) -> ProgressResult:
        pass

    def set_not_promising(self):
        self.promising = False


class Observer(ABC):
    @abstractmethod
    def on_update(self, name: str, progress_result: ProgressResult):
        pass


@dataclass(frozen=True)
class StrategyConfig:
    pass


class StrategyAlgorithm(ABC):
    pass


class Strategy(ABC):
    @abstractmethod
    def execute(self, datasets: "OrderedDict[str, Any]", observer: Observer):
        pass
