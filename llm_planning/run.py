import torch
import hydra

from dataclasses import dataclass

from llm_planning.datasets.strl_robotics import STRLDataset
from llm_planning.gen_methods.full_plan_generation import FullPlanGeneration
from llm_planning.infrastructure.logger import WandbLogger
from torch.utils.data import Subset, random_split
from llm_planning.metrics.lcs import LCSMetrics
from llm_planning.models.llama_7b import LLAMA7B

from hydra.core.config_store import ConfigStore
from llm_planning.processors.strl_processor import STRLProcessor


@dataclass
class LLMPlanningConfig:
    log_filename: str = 'run.log'
    logging_dir: str = "${hydra:run.dir}/"
    seed: int = 0
    device: int = 0
    dataset_path: str = "data/plans.json"


cs = ConfigStore.instance()
cs.store(name="planner", node=LLMPlanningConfig)


@hydra.main(version_base=None, config_path=".", config_name="planner")
def run(cfg=LLMPlanningConfig) -> None:
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    logger = WandbLogger(log_filename=cfg.log_filename,
                         log_dir=cfg.logging_dir,
                         log_to_stdout=True)

    dataset = STRLDataset(cfg.dataset_path)
    train_split, example_split = random_split(
        dataset, [0.8, 0.2], generator=generator)

    processor = STRLProcessor(logger=logger)
    processor.build_system_prompt(example_split)

    model = LLAMA7B(logger=logger,
                    device=cfg.device,
                    name='llama_7b')

    gen_method = FullPlanGeneration(logger=logger,
                                    processor=processor,
                                    model=model)
    lcs_metrics = LCSMetrics(logger=logger,
                             processor=processor)

    for i, gt_task in enumerate(train_split):
        gt_task.text = processor._steps_to_text(gt_task.steps)
        predicted_task = gen_method.predict(gt_task)
        metrics = lcs_metrics.update(predicted_task=predicted_task,
                           target_task=gt_task)
        logger.info(f"Goal:           {gt_task.goal}")
        logger.info(f"GT plan:        {processor._steps_to_text(gt_task.steps)}")
        logger.info(f"Predicted plan: {predicted_task.text}")
        logger.info(f"Metrics:        {metrics}\n\n")

    total_metrics = lcs_metrics.calculate_metrics()
    logger.info(f"Total_metrics:  {total_metrics}")
    logger.info(f"Done.")


if __name__ == "__main__":
    run()
