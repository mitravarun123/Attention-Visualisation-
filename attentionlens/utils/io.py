import json
from pathlib import Path

import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_tensors(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, path)


def load_tensors(path):
    return torch.load(path, map_location="cpu")


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)