from abc import ABC, abstractmethod
from typing import Any


class TaskDataset(ABC):
    data: Any

    def __init__(self) -> None:
        self.get_data()
        
    @abstractmethod
    def get_data(self, *args, **kwargs):
        raise NotImplementedError
    