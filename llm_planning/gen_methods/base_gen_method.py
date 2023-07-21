from abc import ABC, abstractmethod
from typing import Any
from llm_planning.envs.base_env import BaseTask
from llm_planning.infrastructure.logger import BaseLogger

from llm_planning.models.base_model import BaseLLMModel
from llm_planning.processors.base_processor import BaseProcessor


class BasePlanGenerationMethod(ABC):
    """ Base class for plan generation methods. """

    def __init__(self,
                 model: BaseLLMModel,
                 processor: BaseProcessor,
                 logger: BaseLogger,
                 **kwargs):
        self._model = model
        self._prompt_builder = processor
        self._logger = logger
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        """ Prepare model for current method.
        Add step classifier, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: BaseTask, **kwargs) -> Any:
        """ Predict complete plan"""
        raise NotImplementedError
