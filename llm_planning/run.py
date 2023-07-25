import torch
import hydra

from dataclasses import dataclass

from llm_planning.datasets.strl_robotics import STRLDataset
from llm_planning.gen_methods.autoregressive import AutoregressivePlanGeneration
from llm_planning.gen_methods.full import FullPlanGeneration
from llm_planning.infrastructure.config import LLMPlanningConfig
from llm_planning.infrastructure.logger import WandbLogger
from torch.utils.data import Subset, random_split
from llm_planning.metrics.lcs import LCSMetrics
from llm_planning.models.llama_7b import LLAMA7B

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from llm_planning.processors.strl_processor import STRLProcessor



cs = ConfigStore.instance()
cs.store(name="planner", node=LLMPlanningConfig)


@hydra.main(version_base=None, config_path=".", config_name="planner")
def run(cfg=LLMPlanningConfig) -> None:
    generator = torch.Generator()
    generator.manual_seed(cfg.experiment.seed)

    logger: WandbLogger = instantiate(cfg.logger)
    # logger = WandbLogger(log_filename=cfg.log_filename,
    #                      log_dir=cfg.logging_dir,
    #                      log_to_stdout=False)

    # dataset = STRLDataset(cfg.dataset_path)
    dataset = instantiate(cfg.dataset, logger=logger)

    train_split, example_split = random_split(
        dataset, [0.8, 0.2], generator=generator)

    processor = instantiate(cfg.processor, logger=logger)
    # processor = STRLProcessor(logger=logger)
    processor.build_system_prompt(example_split)

    model = instantiate(cfg.model, logger=logger)
    # model = LLAMA7B(logger=logger,
    #                 device=cfg.device,
    #                 name='llama_7b')

    # gen_method = FullPlanGeneration(logger=logger,
    #                                 processor=processor,
    #                                 model=model)
    
    gen_method = instantiate(cfg.gen_method, 
                             logger=logger,
                             processor=processor,
                             model=model)
    # gen_method= AutoregressivePlanGeneration(logger=logger,
    #                                          processor=processor,
    #                                          model=model,
    #                                          saved_steps_path=cfg.saved_steps_path)
    metrics = instantiate(cfg.metrics,
                          logger=logger,
                          processor=processor)
    # lcs_metrics = LCSMetrics(logger=logger,
    #                          processor=processor)

    for i, gt_task in enumerate(train_split):

        gt_task.text = processor._steps_to_text(gt_task.steps)
        predicted_task = gen_method.predict(gt_task)
        curr_metrics = metrics.update(predicted_task=predicted_task,
                           target_task=gt_task)
        logger.info(f"Goal:           {gt_task.goal}")
        logger.info(f"GT plan:        {processor._steps_to_text(gt_task.steps)}")
        logger.info(f"Predicted plan: {predicted_task.text}")
        logger.info(f"Metrics:        {curr_metrics}\n\n")

    total_metrics = metrics.calculate_metrics()
    # total_metrics = {key: f'{value:0.3f}' for key, value in total_metrics.items()}
    logger.info(f"Total_metrics:  {total_metrics}")
    logger.wandb_log(total_metrics)
    logger.info(f"Done.")


if __name__ == "__main__":
    run()
