from dataclasses import InitVar, dataclass, field
from itertools import chain, product
import json
from typing import Any, List, Optional
from pprint import pformat

import torch

from llm_planning.datasets.base_dataset import BaseTask
from llm_planning.infrastructure.logger import BaseLogger
from . import BaseTaskDataset


@dataclass
class Step:
    action: str = ""
    arguments: List[str] = field(default_factory=list)
    text: str = ""
    embedding: Any = field(default_factory=lambda: None,
                           repr=False)


@dataclass
class STRLTask(BaseTask):
    goal: str = ""
    steps: List[Step] = field(default_factory=list)
    text: str = ""
    task_type: int = -1
    plan_id: int = -1
    image: Optional[torch.Tensor] = 1

    def __post_init__(self):
        if self.goal.endswith("."):
            self.goal = self.goal[:-1]


class STRLDataset(BaseTaskDataset):
    def __init__(self, logger: BaseLogger,
                 path_to_dataset: Optional[str] = None):
        self.path_to_dataset = path_to_dataset
        super().__init__(logger=logger)

        with open(path_to_dataset, 'r') as f:
            js = json.load(f)
        self._data = js
        self._size = len(self._data)

        if len(self) == 0:
            raise ValueError("No data")

        self.actions = set()
        self.objects = set()
        self.receptacles = set()

        for item in self:
            for step in item.steps:
                self.actions.add(step.action)
                if len(step.arguments) == 2:
                    self.objects.add(step.arguments[0])
                    self.receptacles.add(step.arguments[1])

        self._logger.info(f'Possible actions:     {self.actions}')
        self._logger.info(f'Possible objects:     {self.objects}')
        self._logger.info(f'Possible receptacles: {self.receptacles}')
        #     for i, step in enumerate(element['plan']):
        #         if step[0] == 'find':
        #             continue
        #         elif step[0] == 'pick_up':
        #             steps.append(['pick_up', step[1][::-1]])
        #         elif step[0] == 'put':
        #             recepticle = element['plan'][i - 1][1][0]
        #             steps.append(['put', step[1] + [recepticle]])
        #         else:
        #             steps.append(step)
        #     element['plan'] = steps

        # for arg_idx, argument in enumerate(step[1:]):
        # pass
        #             if isinstance(argument, list):
        #                 arguments.append(argument[0])
        #             else:
        #                 arguments.append(argument)
        # for element in self._data:
        #     for i, step in enumerate(element['plan']):
        #         output = []
        #         output.append(step[0])
        #         arguments = []
        #         for arg_idx, argument in enumerate(step[1:]):
        #             if isinstance(argument, list):
        #                 arguments.append(argument[0])
        #             else:
        #                 arguments.append(argument)
        #         output.append(arguments)
        #         element['plan'][i] = output
        # with open('out_plan.json' ,'w') as f:
        #     json.dump(self._data, f, ensure_ascii=False)

    def generate_all_possible_steps(self) -> List[Step]:
        possible_steps = []
        for action in self.actions:
            if action == 'put' or action == 'pick_up':
                for obj, recept in product(self.objects, self.receptacles):
                    possible_steps.append(Step(action=action,
                                            arguments=[obj, recept]))
            elif action == 'move_to':
                for target in chain(self.objects, self.receptacles):
                    possible_steps.append(Step(action=action,
                                            arguments=[target]))
        return possible_steps

    def __len__(self):
        return self._size

    def get_data(self):
        pass

    def __getitem__(self, idx) -> STRLTask:
        plan = self._data[idx]
        steps = []
        for step in plan['plan']:
            steps.append(Step(action=step[0],
                              arguments=step[1]))

        return STRLTask(goal=plan['goal_eng'],
                        steps=steps,
                        task_type=plan['task_type'],
                        plan_id=plan["plan_id"])


if __name__ == '__main__':
    dataset = STRLDataset("/home/akorchemnyi/llm_planning/data/plans.json")
    for item in dataset:
        print(item)
    # print(dataset[1])

# {'put', 'pick_up', 'move_to'}
# {'cube', 'cat', 'pepper'}
# {'floor', 'table', 'box', 'chair', 'drawer', 'white box'}
