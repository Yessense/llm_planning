import torch

from llm_planning.infrastructure.logger import WandbLogger
from llm_planning.models.base_model import BaseInput, BaseLLMModel
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import pipeline

from typing import Any
import pprint


class LLAMA7B(BaseLLMModel):
    MODEL_NAME = "decapoda-research/llama-7b-hf"

    def __init__(self,
                 logger: WandbLogger,
                 device: int = 2,
                 name: str = 'llama_7b') -> None:
        self.max_new_tokens = 45
        self.device = device
        super().__init__(name=name, logger=logger)
        # self.logger.info(f"Device map: \n{pprint.pformat(device_map)}")

    # def score_text(self, **kwargs) -> Any:
    #     pass
    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "decapoda-research/llama-7b-hf",
            torch_dtype=torch.float16,
            # load_in_8bit=True,
            device_map={'': self.device},
        )
        self.model.eval()

        self._logger.info(
            f"Model device map: \n{pprint.pformat(self.model.hf_device_map)}")

        self.tokenizer = LlamaTokenizer.from_pretrained(self.MODEL_NAME)
        self._prepare_for_generation()


    def _prepare_for_generation(self) -> None:
        self.generation_pipeline = pipeline("text-generation",
                                            model=self.model,
                                            tokenizer=self.tokenizer)

    def generate(self, inputs: BaseInput, **kwargs) -> str:
        # generate plan
        output = self.generation_pipeline(inputs.text,
                                          do_sample=False,
                                          return_full_text=False,
                                          max_new_tokens=self.max_new_tokens)
        return output[0]['generated_text']

if __name__ == "__main__":
    wandb_logger = WandbLogger(log_filename=None)
    model = LLAMA7B(name='llama_7b', logger=wandb_logger)

    answer = model.generate("questin")

    print(answer)