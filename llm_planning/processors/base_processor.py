from abc import ABC, abstractmethod
from llm_planning.datasets.base_dataset import BaseTask
from llm_planning.infrastructure.logger import BaseLogger

from llm_planning.models.base_model import BaseInput, BaseOutput


class BaseProcessor(ABC):
    def __init__(self, logger: BaseLogger, **kwargs):
        self._logger = logger


    @abstractmethod
    def to_inputs(self, task: BaseTask) -> BaseInput:
        raise NotImplementedError

    @abstractmethod
    def to_task(self, task: BaseOutput) -> BaseTask:
        raise NotImplementedError


