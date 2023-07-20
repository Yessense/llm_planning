from abc import ABC
from typing import Optional

from llm_planning.infrastructure.logger import PlanningLogger


class BaseLLMModel(ABC):
    """Base class for LLM models"""

    _name: str
    _logger: Optional[PlanningLogger]