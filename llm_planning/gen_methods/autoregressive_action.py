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

# ACTIONS = ['put', 'pick_up', 'move_to']
# OBJECTS = ['cube', 'cat', 'pepper']
# RECEPTICLES = ['floor', 'table', 'box', 'chair', 'drawer', 'white box']

# def generate_all_possible_steps(actions: List[str],
#                                 objects: List[str],
#                                 recepticles: List[str]) -> List[Step]:
#     possible_steps = []
#     for action in actions:
#         if action == 'put' or action == 'pick_up':
#             for obj, recept in product(objects, recepticles):
#                 possible_steps.append(Step(action=action,
#                                            arguments=[obj, recept]))
#         elif action == 'move_to':
#             for target in chain(objects, recepticles):
#                 possible_steps.append(Step(action=action,
#                                            arguments=[target]))
#     return possible_steps


class StepClassifier:
    _embed_dim: int = 384

    @property
    def embed_dim(self):
        return self._embed_dim

    def __init__(self,
                 processor: STRLProcessor,
                 saved_steps_path: Optional[str],
                 dataset: BaseTaskDataset,
                 ) -> None:

        self._model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self._processor = processor

        # load or save all steps with embeddings in pickle
        if os.path.isfile(saved_steps_path):
            with open(saved_steps_path, 'rb') as f:
                self._all_possible_actions, self.embeddings = pickle.load(f)
        else:
            self._all_possible_actions = list(dataset.actions)
            self._calculate_embeddings(self._all_possible_actions)
            # print(self._all_possible_steps)

            with open(saved_steps_path, 'wb') as f:
                pickle.dump((self._all_possible_actions, self.embeddings), f)


    def get_closest_step(self, step: Step) -> Step:
        if not len(step.text):
            step.text = self._processor._step_to_text(step)
        assert len(step.text)

        embedding = self._model.encode(step.action)

        cosine_sims = cosine_similarity([embedding], self.embeddings)
        closest_emb_idx = np.argmax(cosine_sims)

        action = self._all_possible_actions[closest_emb_idx]
        # print(f'Target step:    {step.text}')
        # print(f'Predicted_step: {closest_step.text}')
        step.action = action
        return step
        
    def _calculate_embeddings(self, actions: List[str]):
        self.embeddings = self._model.encode(actions)


class AutoregressivePlanGenerationAction(BasePlanGeneration):
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
        self._step_classifier = StepClassifier(processor=self._processor,
                                               saved_steps_path=self._saved_steps_path,
                                               dataset=dataset)

    def setup(self) -> None:
        pass
                                             
        
    def predict(self, gt_task: BaseTask) -> BaseTask:
        steps: List[Step] = []
        while len(steps) <= self._max_plan_length:
            inputs = self._processor.to_inputs(gt_task, steps)
            model_ouputs = self._model.generate(inputs)
            output_step = self._processor._text_to_steps(model_ouputs.text, cut_one_step=True)
            if output_step is None or self._processor.is_terminating(output_step):
                break
            else:
                closest_step = self._step_classifier.get_closest_step(output_step)
                steps.append(closest_step)
        return self._processor.to_task(steps)

if __name__ == "__main__":
    # all_possible_steps = generate_all_possible_steps(actions=ACTIONS,
    #                                                  objects=OBJECTS,
    #                                                  recepticles=RECEPTICLES)
    # pprint(all_possible_steps)
    logger = WandbLogger('test_autoregressive')
    processor = STRLProcessor(logger)
    classifier = StepClassifier(processor, '/home/akorchemnyi/llm_planning/data/all_possible_steps.pkl')
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

    
