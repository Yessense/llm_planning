import logging
import wandb
import os
from abc import ABC, abstractmethod


class PlanningLogger(ABC):
    _name: str
    _project_name: str

    def __init__(self,
                 log_dir: str,
                 project_name: str,
                 log_filename: str):
        self._log_dir = log_dir
        self._log_filename = os.path.join(self._log_dir, log_filename)
        self._logger = logging.getLogger(self._name)

    @abstractmethod
    def log_text(self, text: str):
        raise NotImplementedError


class WandbLogger(PlanningLogger):
    def __init__(self, log_dir: str, project_name: str, log_filename: str):
        

    super().__init__(log_dir, project_name, log_filename)
    def log_text(self, text: str):
        pass