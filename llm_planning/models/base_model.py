from abc import ABC, abstractmethod
from typing import Any, Optional

from llm_planning.infrastructure.logger import PlanningLogger


    
class BaseLLMModel(ABC):
    """ Base class for LLM models"""
    _name: str
    _logger: PlanningLogger

    @property
    def name(self):
        return self._name

    def __init__(self,
                 name: str,
                 logger: PlanningLogger,
                 **kwargs) -> None:
        self._name = name
        self._logger = logger

        self._logger.info(f"Loading {name} model...")
        self._load()
        self._logger.info(f'Model: {name}\n')

    def _prepare_for_generation(self) -> None:
        """Define pipeline, etc."""
        return None

    # def _prepare_for_scoring(self) -> None:
    #     """Define scoring for saycan mode"""
    #     pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """ Generate text"""
        pass

    # @abstractmethod
    # def score_text(self, **kwargs) -> Any:
    #     """ Score text for saycan approach """
    #     pass

    @abstractmethod
    def _load(self) -> None:
        """Load model, tokenizer, etc"""
        pass