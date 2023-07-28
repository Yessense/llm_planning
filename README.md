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

## Language table dataset exploration