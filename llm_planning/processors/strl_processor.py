import re
from typing import List, Optional, Union

import torch
from llm_planning.datasets.strl_robotics import STRLDataset, STRLTask, Step
from llm_planning.infrastructure.logger import BaseLogger
from llm_planning.models.base_model import BaseInput, BaseOutput, ScoringInput
from llm_planning.models.minigpt4 import Minigpt4Input
from llm_planning.processors.base_processor import BaseProcessor


class STRLProcessor(BaseProcessor):
    TERMINATING_STRING = 'done()'

    @property
    def system_prompt_is_set(self) -> bool:
        return len(self._system_prompt) > 0

    def is_terminating(self, step: Step) -> bool:
        return step.text == self.TERMINATING_STRING

    def __init__(self, logger: BaseLogger, **kwargs) -> None:
        super().__init__(logger=logger, **kwargs)
        self._system_prompt = ""
        self._stop_step_pattern = ""
        self._stop_pattern = re.compile(f'\\d+\\. {self.TERMINATING_STRING}.')

    def build_system_prompt(self, example_tasks: List[STRLTask]) -> str:
        prompt = "Robot: Hi there, I’m a robot operating in a house.\n"
        prompt += "Robot: You can ask me to do various tasks and "
        prompt += "I’ll tell you the sequence of actions I would do to accomplish your task.\n"

        for task in example_tasks:
            prompt += self._task_to_prompt(task) + '\n'

        self._system_prompt = prompt
        self._stop_step_pattern = re.compile(
            r'(\s*\d+\.\s*)(\w+\(("[\w ]+"(,\s)?)*\))*')
        self._logger.info("Building system prompt...")
        self._logger.info("\n" + self._system_prompt)

    def _goal_to_query(self, goal: str) -> str:
        query = f"Human: How would you {goal.lower()}?\n"
        query += f'Robot: '
        return query

    def _step_to_text(self, step: Step) -> str:
        arguments = [f'"{argument}"' for argument in step.arguments]
        text = f'{step.action}({", ".join(arguments)})'
        return text

    def _steps_to_text(self,
                       steps: List[Step],
                       add_terminating_string: bool = True) -> str:
        text = ", ".join([f'{step_idx}. {self._step_to_text(step)}'
                          for step_idx, step in enumerate(steps, start=1)])
        if add_terminating_string:
            text += f", {len(steps) + 1}. {self.TERMINATING_STRING}."
        return text

    def _task_to_prompt(self, task: STRLTask) -> str:
        prompt = self._goal_to_query(task.goal)
        text = self._steps_to_text(task.steps)
        task.text = text
        prompt += text
        return prompt

    def to_inputs(self,
                  task: STRLTask,
                  steps: Optional[List[Step]] = None,
                  options: Optional[List[Step]] = None) -> BaseInput:
        if not self.system_prompt_is_set:
            raise ValueError(
                "System prompt is not set. You need to set system prompt.")
        else:
            text = self._system_prompt + self._goal_to_query(task.goal)
            if steps is not None:
                text += self._steps_to_text(steps, add_terminating_string=False)
            if options is not None:
                return ScoringInput(text=text, options=[f'{len(steps) + 1}. {option.text}' for option in options])
            if task.image is not None:
                # TODO: fix this
                return Minigpt4Input(text=text, image=torch.zeros(3,224,224))
            return BaseInput(text=text)

    def _text_to_steps(self, task_text: str, cut_one_step: bool = False) -> Union[List[Step], Step, None]:
        if cut_one_step:
            stop_match = self._stop_step_pattern.match(task_text)
            if stop_match is None:
                return None
            else:
                return self._parse_action(stop_match.group(2))
        else:
            stop_match = self._stop_step_pattern.findall(task_text)
            steps = []
            if stop_match is None:
                return steps
            else:
                for i in range(len(stop_match) - 1):
                    step_text = stop_match[i][1]
                    step = self._parse_action(step_text)
                    if step is not None:
                        steps.append(step)
                return steps

    def _parse_action(self, step_text: str) -> Optional[Step]:
        """ Parse action with arguments to step.
        text: put_on('pepper', 'white box')
        action: put_on
        arguments: ['pepper', 'white box']
        """
        step_decomposition_pattern = re.compile(r'\s*([A-Za-z_][A-Za-z_\s]+)')
        arguments = step_decomposition_pattern.findall(step_text)

        if arguments is None:
            return None
        if len(arguments) == 1:
            step = Step(text=step_text)
        else:
            step = Step(action=arguments[0],
                        arguments=arguments[1:],
                        text=step_text)
            return step

    def to_task(self, task: Union[BaseOutput, List[Step]]) -> STRLTask:
        # Autoregressive Mode
        if isinstance(task, list):
            return STRLTask(text=self._steps_to_text(task),
                            steps=task)
        
        # Full plan generation mode
        stop_match = self._stop_pattern.search(task.text)

        if stop_match is not None:
            task.text = task.text[:stop_match.end() + 2].strip(' \n\t')
        else:
            task.text = task.text.strip(' \n\t')

        steps = self._text_to_steps(task_text=task.text)

        return STRLTask(text=task.text, steps=steps)


if __name__ == "__main__":
    dataset = STRLDataset("data/plans.json")

    processor = STRLProcessor()
    processor.build_system_prompt([dataset[0], dataset[3]])

    print(processor.to_inputs(dataset[1]))
