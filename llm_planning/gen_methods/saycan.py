from itertools import chain, product
import pickle
from typing import Dict, List, Optional
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity
from llm_planning.datasets import base_dataset
from llm_planning.datasets.strl_robotics import Step
from pprint import pprint

from llm_planning.gen_methods.base_gen import BasePlanGeneration
from llm_planning.infrastructure.logger import BaseLogger, WandbLogger
from llm_planning.models.base_model import BaseLLMModel, BaseInput
from llm_planning.datasets.base_dataset import BaseTask, BaseTaskDataset
from llm_planning.processors.base_processor import BaseProcessor
from sentence_transformers import SentenceTransformer

from llm_planning.processors.strl_processor import STRLProcessor


class SaycanPlanGeneration(BasePlanGeneration):
    def __init__(self,
                 model: BaseLLMModel,
                 processor: STRLProcessor,
                 logger: BaseLogger,
                 saved_steps_path: str,
                 dataset: BaseTaskDataset,
                 max_plan_length: int = 10,
                 **kwargs):
        self._saved_steps_path = saved_steps_path
        self._max_plan_length = max_plan_length
        super().__init__(model, processor, logger, **kwargs)

    def setup(self) -> None:
        pass

    def predict(self, gt_task: BaseTask) -> BaseTask:
        steps: List[Step] = []
        while len(steps) <= self._max_plan_length:
            inputs = self._processor.to_inputs(gt_task, steps)
            model_ouputs = self._model.generate(inputs)
            output_step = self._processor._text_to_steps(
                model_ouputs.text, cut_one_step=True)
            if output_step is None or self._processor.is_terminating(output_step):
                break
            else:
                closest_step = self._step_classifier.get_closest_step(
                    output_step)
                steps.append(closest_step)
        return self._processor.to_task(steps)


if __name__ == "__main__":
    # all_possible_steps = generate_all_possible_steps(actions=ACTIONS,
    #                                                  objects=OBJECTS,
    #                                                  recepticles=RECEPTICLES)
    # pprint(all_possible_steps)
    logger = WandbLogger('test_autoregressive')
    processor = STRLProcessor(logger)
    print(f'1. Example. Семантическая близость (кошка - собака)')
    test_step = Step(action='pick_up',
                     arguments=['dog', 'box'])
    answer = classifier.get_closest_step(test_step)
    print(f'2. Example. Самое вероятное вместилище для контейнера')
    test_step = Step(action='pick_up',
                     arguments=['pepper', 'container'])
    answer = classifier.get_closest_step(test_step)
    print(f'3. Example. Другой цвет')
    test_step = Step(action='pick_up',
                     arguments=['pepper', 'red box'])
    answer = classifier.get_closest_step(test_step)
    print(f'4. Example. Самая вероятная игрушка')
    test_step = Step(action='pick_up',
                     arguments=['toy', 'floor'])
    answer = classifier.get_closest_step(test_step)
    print(f'5. Example. Шкаф -> Ящик')
    test_step = Step(action='pick_up',
                     arguments=['toy', 'cabinet'])
    answer = classifier.get_closest_step(test_step)
