from attentionlens.utils.io import load_yaml


def load_prompts(path="configs/prompts.yaml"):
    return load_yaml(path)["prompts"]