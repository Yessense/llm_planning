from typing import Dict, List

from llm_planning.gen_methods.base_gen import BasePlanGeneration
from llm_planning.infrastructure.logger import BaseLogger, WandbLogger
from llm_planning.models.base_model import BaseLLMModel, BaseInput
from llm_planning.datasets.base_dataset import BaseTask
from llm_planning.processors.base_processor import BaseProcessor


class FullPlanGeneration(BasePlanGeneration):
    def __init__(self,
                 model: BaseLLMModel,
                 processor: BaseProcessor,
                 logger: BaseLogger,
                 **kwargs):
        super().__init__(model, processor, logger, **kwargs)

    def setup(self) -> None:
        pass

    def predict(self, gt_task: BaseTask) -> BaseTask:
        inputs = self._processor.to_inputs(gt_task)
        model_ouputs = self._model.generate(inputs)
        output_task = self._processor.to_task(model_ouputs)
        return output_task