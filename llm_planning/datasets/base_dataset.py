from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List
import torch
from torch.utils.data import Dataset

from llm_planning.infrastructure.logger import BaseLogger



@dataclass
class BaseTask(ABC):
    pass


class BaseTaskDataset(ABC, Dataset):
    data: Any

    def __init__(self, logger: BaseLogger) -> None:
        self._logger = logger
        self.get_data()

        
    @abstractmethod
    def get_data(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx) -> BaseTask:
        raise NotImplementedError

    @abstractmethod
    def generate_all_possible_steps(self) -> List:
        raise NotImplementedError
    