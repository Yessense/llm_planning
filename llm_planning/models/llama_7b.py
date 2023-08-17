import torch
from tqdm import tqdm
from llm_planning.datasets.strl_robotics import STRLDataset, STRLTask, Step
from llm_planning.gen_methods.full import FullPlanGeneration

from llm_planning.infrastructure.logger import WandbLogger
from llm_planning.models.base_model import BaseInput, BaseLLMModel, BaseOutput, ScoringInput, ScoringOutput
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import pipeline
import torch.nn.functional as F

from typing import Any, List
import pprint

from llm_planning.processors.strl_processor import STRLProcessor


class LLAMA7B(BaseLLMModel):
    MODEL_NAME = "decapoda-research/llama-7b-hf"

    def __init__(self,
                 logger: WandbLogger,
                 device: int = 1,
                 name: str = 'llama_7b',
                 max_new_tokens: int = 100) -> None:
        self.max_new_tokens = max_new_tokens
        self.device = device
        super().__init__(name=name, logger=logger)
        # self.logger.info(f"Device map: \n{pprint.pformat(device_map)}")

    # def score_text(self, **kwargs) -> Any:
    #     pass
    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "decapoda-research/llama-7b-hf",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            # device_map='auto',
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

    def generate(self, inputs: BaseInput, **kwargs) -> BaseOutput:
        # generate plan
        output = self.generation_pipeline(inputs.text,
                                          do_sample=False,
                                          return_full_text=False,
                                          max_new_tokens=self.max_new_tokens)
        output = BaseOutput(output[0]['generated_text'])
        return output
    
    def score_text(self,
                   inputs: ScoringInput,
                   option_start: str = '\n',
                   **kwargs) -> Any:
        scores = torch.zeros(len(inputs.options))
        
        for i, option in enumerate(tqdm(inputs.options)):
            score = self.score_option(query=inputs.text, option=option)
            scores[i] = score
            # token_texts.append(token_txt)
            # token_probabilities.append(token_probs)
        return ScoringOutput(scores=scores)


    def score_option(self, query, option):
        text = query + option
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        # inputs_tokenized = [self.tokenizer.decode(token) for token in inputs.input_ids[0]]
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            # torch.Tensor of shape (1, 843, 32000)
            #                       (batch_size, n_tokens, vocab_dim)
            logits = outputs.logits
        
        # squeeze() to remove batch dimension
        # cut off the last token, because it refers to the next generated character
        predictions = F.log_softmax(logits, dim=-1).squeeze()[:-1]
        # predictions -> (842, 32000)

        # input_ids -> (1, 843)
        # tokens with start token (0, 6417, 327, ...)        
        # cut start token
        input_ids = inputs.input_ids.squeeze(0)[1:]
        gen_probs = torch.gather(predictions, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        
        # text_token = []
        # text_token_logprob = []
        # to stop gathering prob
        option_tokenized = self.tokenizer(option, return_tensors="pt").input_ids[0]
        # option_as_tokens = [self.tokenizer.decode(token) for token in option_tokenized]
        score = 0

        for i, (token, prob) in enumerate(zip(reversed(input_ids), reversed(gen_probs))):
            # break when option_ends
            if i == len(option_tokenized) - 2:
                break
            # text_token.append(self.tokenizer.decode(token))
            # text_token_logprob.append(prob.item())
            score += prob.item()
    
        return score #text_token, text_token_logprob


if __name__ == "__main__":
    wandb_logger = WandbLogger(log_filename='test_llama')
    model = LLAMA7B(name='llama_7b', logger=wandb_logger)
    dataset = STRLDataset(logger=wandb_logger, path_to_dataset='/home/akorchemnyi/llm_planning/data3/new_plans_with_args.json')
    processor = STRLProcessor(wandb_logger)
    processor.build_system_prompt(dataset)

    gen_method = FullPlanGeneration(model, processor, wandb_logger)

    task = STRLTask(text = "Move a red apple from the table to the drawer.")
    predicted_task = gen_method.predict(gt_task=task)

    print(predicted_task.text)

