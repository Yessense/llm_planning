# LLM-planning
Framework for evaluating planning with LLMs.

## How to install

```bash
git clone https://github.com/Yessense/llm_planning.git
conda env create -f environment.yaml
# uncomment next line to login in wandb
# wandb login
```

## How to run experiment

```bash
python3 -m llm_planning.run
```

You may change run parameters in `llm_planning.infrastructure.config.py` write them down in `config.config.yaml` or add them as a command line arguments.

## Language table dataset

Dataset notebook <a target="_blank" href="https://colab.research.google.com/github/Yessense/llm_planning/blob/master/language_table/Dataset.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Data exploration notebook <a target="_blank" href="https://colab.research.google.com/github/Yessense/llm_planning/blob/master/speech_recognition/Speech%20recognition%20in%20noise%20Dolgushin.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Speech recognition in noisy environments

Notebook with data processing and wav2vec model finetuning and evaluation whisper and wav2vec models <a target="_blank" href="https://colab.research.google.com/github/Yessense/llm_planning/blob/master/speech_recognition/Speech%20recognition%20in%20noise%20Dolgushin.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

