from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from omegaconf import MISSING


@dataclass
class BaseLoggerConfig:
    _target_: str = MISSING


@dataclass
class BaseDatasetConfig:
    _target_: str = MISSING
    path_to_dataset: str = MISSING


@dataclass
class BaseModelConfig:
    name: str = MISSING
    _target_: str = MISSING


@dataclass
class BaseMetricsConfig:
    _target_: str = MISSING


@dataclass
class LCSMetricsConfig(BaseMetricsConfig):
    _target_: str = "llm_planning.metrics.lcs.LCSMetrics"


@dataclass
class BasePlanGenConfig:
    _target_: str = MISSING
    name: str = MISSING


@dataclass
class BaseProcessorConfig:
    _target_: str = MISSING
    name: str = MISSING


@dataclass
class BaseExperimentConfig:
    device: int = 2
    logging_dir: str = "${hydra:run.dir}/"
    seed: int = 0
    path_to_data_dir: str = "${hydra:runtime.cwd}/data/"


# Loggers
@dataclass
class WandbLoggerConfig(BaseLoggerConfig):
    _target_: str = "llm_planning.infrastructure.logger.WandbLogger"
    log_filename: str = 'run.log'
    log_dir: str = "${hydra:run.dir}/"
    project_name: str = "llm_planning"
    run_name: str = "${model.name} ${gen_method.name}"


# Plan generation
@dataclass
class FullPlanGenerationConfig(BasePlanGenConfig):
    _target_: str = "llm_planning.gen_methods.full.FullPlanGeneration"
    name: str = "full generation"


@dataclass
class AutoregressivePlanGenerationConfig(BasePlanGenConfig):
    _target_: str = "llm_planning.gen_methods.autoregressive.AutoregressivePlanGeneration"
    name: str = "autoregressive generation"
    max_plan_size: int = 10
    saved_steps_path: str = "${experiment.path_to_data_dir}/all_possible_steps.pkl"


# Datasets
@dataclass
class STRLDatasetConfig(BaseDatasetConfig):
    path_to_dataset: str = '${experiment.path_to_data_dir}/plans.json'
    _target_: str = 'llm_planning.datasets.strl_robotics.STRLDataset'


# Models
@dataclass
class LLAMA7BModelConfig(BaseModelConfig):
    _target_: str = 'llm_planning.models.llama_7b.LLAMA7B'
    device: int = "${experiment.device}"
    name: str = 'llama7b'
    max_new_tokens: int = 100


@dataclass
class STRLProcessorConfig(BaseProcessorConfig):
    _target_: str = "llm_planning.processors.strl_processor.STRLProcessor"
    name: str = "strl_processor"


@dataclass
class LLMPlanningConfig:
    model: BaseModelConfig = field(default_factory=LLAMA7BModelConfig)
    gen_method: BasePlanGenConfig = field(default_factory=FullPlanGenerationConfig)
    experiment: BaseExperimentConfig = field(default_factory=BaseExperimentConfig)
    dataset: BaseDatasetConfig = field(default_factory=STRLDatasetConfig)
    logger: BaseLoggerConfig = field(default_factory=WandbLoggerConfig)
    metrics: BaseMetricsConfig = field(default_factory=LCSMetricsConfig)
    processor: BaseProcessorConfig = field(default_factory=STRLProcessorConfig)
