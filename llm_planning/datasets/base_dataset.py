from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch
from torch.utils.data import Dataset



@dataclass
class BaseTask(ABC):
    pass


class BaseTaskDataset(ABC, Dataset):
    data: Any

    def __init__(self) -> None:
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
    