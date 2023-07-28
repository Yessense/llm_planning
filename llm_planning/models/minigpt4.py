from dataclasses import dataclass
from typing import Optional
import torch
from llm_planning.datasets.strl_robotics import Step

from llm_planning.infrastructure.logger import WandbLogger
from llm_planning.models.base_model import BaseInput, BaseLLMModel, BaseOutput, ScoringInput, ScoringOutput
import torch.nn.functional as F


import json
import pathlib
import huggingface_hub
import os
import gdown
import yaml
import sys
import torch
import argparse 
from transformers import StoppingCriteriaList
from PIL import Image

@dataclass
class Minigpt4Input(BaseInput):
    image: Optional[str] = None


class MiniGPT4(BaseLLMModel):
    def __init__(self,
                 logger: WandbLogger,
                 device: int = 2,
                 name: str = 'minigpt4',
                 max_new_tokens: int = 200) -> None:
        self._num_beams = 1
        self._temperature = 0.9
        self._max_new_tokens = max_new_tokens
        self.device = device
        super().__init__(name=name, logger=logger)

    def score_text(self):
        pass

    def _load(self):
        default_cache_dir = ".cache"
        # clone repo 
        os.system(f"git clone https://github.com/Vision-CAIR/MiniGPT-4.git {default_cache_dir}/MiniGPT-4")
        # download models
        llama_space = "decapoda-research"
        llama_id = "llama-7b-hf"
        vicuna_space = "lmsys"
        vicuna_id = "vicuna-7b-delta-v0"
        llama_repo_id = f"{llama_space}/{llama_id}"
        vicuna_repo_id = f"{vicuna_space}/{vicuna_id}"
        huggingface_hub.snapshot_download(repo_id=llama_repo_id, cache_dir=default_cache_dir)
        huggingface_hub.snapshot_download(repo_id=vicuna_repo_id, cache_dir=default_cache_dir)

        for space, repo in [(vicuna_space, vicuna_id), (llama_space, llama_id)]:
            for path in pathlib.Path(f"{default_cache_dir}/models--{space}--{repo}/snapshots/").rglob("*/tokenizer_config.json"):
                print(f"Loading {path}")
                config = json.loads(open(path, "r").read())
                if config["tokenizer_class"] == "LlamaTokenizer":
                    print("No fix needed")
                else:
                    config["tokenizer_class"] = "LlamaTokenizer"
                with open(path, "w") as f:
                    json.dump(config, f)

        # if not os.path.exists(f"{default_cache_dir}/vicuna-7b-v0"):
        os.system(f"poetry run python3 -m fastchat.model.apply_delta --base-model-path {default_cache_dir}/models--{llama_space}--{llama_id}/snapshots/*/ --target-model-path {default_cache_dir}/vicuna-7b-v0 --delta-path {default_cache_dir}/models--{vicuna_space}--{vicuna_id}/snapshots/*/")
        output_path = f'{default_cache_dir}/pretrained_minigpt4.pth'
        gdown.download(
            "https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing", output_path, fuzzy=True
        )

        eval_config_path = pathlib.Path(f"{default_cache_dir}/MiniGPT-4/eval_configs/minigpt4_eval.yaml")
        with open(eval_config_path, "r") as f:
            eval_config_dict = yaml.safe_load(f)
            eval_config_dict["model"]["ckpt"] = f"{default_cache_dir}/pretrained_minigpt4.pth"
            eval_config_dict["model"]["prompt_path"] = f"{default_cache_dir}/MiniGPT-4/prompts/alignment.txt"
            
        with open(eval_config_path, "w") as f:
            yaml.dump(eval_config_dict, f)

        minigpt4_config_path = pathlib.Path(f"{default_cache_dir}/MiniGPT-4/minigpt4/configs/models/minigpt4.yaml")
        with open(minigpt4_config_path, "r") as f:
            minigpt4_config_dict = yaml.safe_load(f)
            minigpt4_config_dict["model"]["llama_model"] = f"{default_cache_dir}/vicuna-7b-v0"
            
        with open(minigpt4_config_path, "w") as f:
            yaml.dump(minigpt4_config_dict, f)

        minigpt4_path = f'{default_cache_dir}/MiniGPT-4'
            
        if sys.path[-1] != minigpt4_path:
            sys.path.append(minigpt4_path)

        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import StoppingCriteriaSub, Conversation, SeparatorStyle

        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--cfg-path', help='')
        parser.add_argument('--options', nargs="+",help='')
        parser.add_argument('--gpu-id', default=0, help='')
        args = parser.parse_args(f" --cfg-path {default_cache_dir}/MiniGPT-4/eval_configs/minigpt4_eval.yaml".split())

        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train


        class MiniGPT4Chat:
            
            def __init__(self, model, vis_processor, device='cuda:0'):
                self.device = device
                self.model = model
                self.vis_processor = vis_processor
                stop_words_ids = [torch.tensor([835]).to(self.device),
                                torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
                self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
                self.conv, self.img_list = None, None
                self.reset_history()
                
            def ask(self, text):
                if len(self.conv.messages) > 0 and self.conv.messages[-1][0] == self.conv.roles[0] \
                        and self.conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
                    self.conv.messages[-1][1] = ' '.join([self.conv.messages[-1][1], text])
                else:
                    self.conv.append_message(self.conv.roles[0], text)

            def answer(self, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                    repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
                self.conv.append_message(self.conv.roles[1], None)
                embs = self.get_context_emb(self.img_list)

                current_max_len = embs.shape[1] + max_new_tokens
                if current_max_len - max_length > 0:
                    print('Warning: The number of tokens in current conversation exceeds the max length. '
                        'The model will not see the contexts outside the range.')
                begin_idx = max(0, current_max_len - max_length)

                embs = embs[:, begin_idx:]

                outputs = self.model.llama_model.generate(
                    inputs_embeds=embs,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=self.stopping_criteria,
                    num_beams=num_beams,
                    do_sample=True if num_beams==1 else False,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    temperature=temperature,
                )
                output_token = outputs[0]
                if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                    output_token = output_token[1:]
                if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                    output_token = output_token[1:]
                output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
                output_text = output_text.split('###')[0]  # remove the stop sign '###'
                output_text = output_text.split('Assistant:')[-1].strip()
                self.conv.messages[-1][1] = output_text
                return output_text, output_token.cpu().numpy()

            def upload_img(self, image):
                if isinstance(image, str):  # is a image path
                    raw_image = Image.open(image).convert('RGB')
                    image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
                elif isinstance(image, Image.Image):
                    raw_image = image
                    image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
                elif isinstance(image, torch.Tensor):
                    if len(image.shape) == 3:
                        image = image.unsqueeze(0)
                    image = image.to(self.device)

                image_emb, _ = self.model.encode_img(image)
                self.img_list.append(image_emb)
                self.conv.append_message(self.conv.roles[0], "<Img><ImageHere></Img>")
                msg = "Received."
                return msg

            def get_context_emb(self, img_list):
                prompt = self.conv.get_prompt()
                prompt_segs = prompt.split('<ImageHere>')
                assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
                seg_tokens = [
                    self.model.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                    # only add bos to the first seg
                    for i, seg in enumerate(prompt_segs)
                ]
                seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
                mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
                mixed_embs = torch.cat(mixed_embs, dim=1)
                return mixed_embs
            
            def reset_history(self):
                self.conv = Conversation(
                    system="Give the following image: <Img>ImageContent</Img>. "
                        "You will be able to see the image once I provide it to you. Please answer my questions.",
                    roles=("Human", "Assistant"),
                    messages=[],
                    offset=2,
                    sep_style=SeparatorStyle.SINGLE,
                    sep="###",
                )
                self.img_list = []
    
        self._model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        self._model.eval()
        self._vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.minigpt4 = MiniGPT4Chat(self._model, self._vis_processor)


    def generate(self, inputs: Minigpt4Input) -> BaseOutput:
        self.minigpt4.reset_history()
        
        self.minigpt4.upload_img(inputs.image)
        self.minigpt4.ask(inputs.text)
        out, _ = self.minigpt4.answer(
            num_beams=self._num_beams,
            temperature=self._temperature,
            max_new_tokens=self._max_new_tokens,
        )    
        return BaseOutput(out)
            
            
if __name__ == "__main__":
    wandb_logger = WandbLogger(log_filename='test_minigpt4')
    minigpt4 = MiniGPT4(logger=wandb_logger)

    inputs = Minigpt4Input(image="./images/img.jpeg",
                            text="Look on this image and describe it")
    minigpt4.generate(inputs)


    answer = minigpt4.generate(inputs)

    print(answer)