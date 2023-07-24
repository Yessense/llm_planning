import re
from typing import List, Optional
from llm_planning.datasets.strl_robotics import STRLDataset, STRLTask, Step
from llm_planning.infrastructure.logger import BaseLogger
from llm_planning.models.base_model import BaseInput, BaseOutput
from llm_planning.processors.base_processor import BaseProcessor


class STRLProcessor(BaseProcessor):
    TERMINATING_STRING = 'done()'

    @property
    def system_prompt_is_set(self):
        return len(self._system_prompt)

    def __init__(self, logger: BaseLogger) -> None:
        super().__init__(logger=logger)
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
        self._stop_step_pattern = re.compile(r'(\s*\d+\.\s*)*(.*?)(?=\.|,)')
        self._logger.info("Building system prompt...")
        self._logger.info("\n" + self._system_prompt)

    def _goal_to_query(self, goal: str) -> str:
        query = f"Human: How would you {goal.lower()}?\n"
        query += f'Robot: '
        return query

    def _step_to_text(self, step: Step) -> str:
        text = f'{step.action}({", ".join([repr(argument) for argument in step.arguments])})'
        return text

    def _steps_to_text(self, steps: List[Step]) -> str:
        text = ", ".join([f'{step_idx}. {self._step_to_text(step)}' 
                          for step_idx, step in enumerate(steps, start=1)])
        text += f", {len(steps) + 1}. {self.TERMINATING_STRING}."

        return text
    
    def _task_to_prompt(self, task: STRLTask) -> str:
        prompt = self._goal_to_query(task.goal)
        text = self._steps_to_text(task.steps)
        task.text = text
        prompt += text
        return prompt

    def to_inputs(self, task: STRLTask) -> BaseInput:
        if not self.system_prompt_is_set:
            raise ValueError("System prompt is not set. You need to set system prompt.")
        else:
            return BaseInput(text=self._system_prompt + self._goal_to_query(task.goal))

    def cut_step_from_generated_text(self, generated_text: str) -> Optional[str]:
        stop_match = self.stop_step_pattern.search(generated_text)

        if stop_match is not None:
            if stop_match.groups()[-1].isnumeric():
                return None
            else:
                return stop_match.groups()[-1]
        else:
            return None

    def to_task(self, task: BaseOutput) -> STRLTask:
        stop_match = self._stop_pattern.search(task.text)

        if stop_match is not None:
            task.text=task.text[:stop_match.end() + 2].strip(' \n\t')
        else:
            task.text=task.text.strip(' \n\t')

        return STRLTask(text=task.text)
        

if __name__ == "__main__":
    dataset = STRLDataset("data/plans.json")

    processor = STRLProcessor()
    processor.build_system_prompt([dataset[0], dataset[3]])

    print(processor.to_inputs(dataset[1]))
    