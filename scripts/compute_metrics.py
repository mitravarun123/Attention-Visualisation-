from pathlib import Path

import torch

from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.head_types import classify, head_scores
from attentionlens.utils.io import load_tensors, load_yaml, save_json

CAUSAL = {"gpt2": True, "bert": False, "retnet": True, "rwkv": True}


def main():
    cfg = load_yaml("configs/classifier.yaml")
    for path in sorted(Path("outputs/tensors").glob("*.pt")):
        name = path.stem
        ents, scs = [], []
        for rec in load_tensors(path):
            attn = rec["attn"].float()
            ents.append(head_entropy(attn, CAUSAL[name]))
            scs.append(head_scores(attn, cfg["window"]))
        ent = torch.stack(ents).mean(0)
        scores = {k: torch.stack([s[k] for s in scs]).mean(0) for k in scs[0]}
        types = classify(scores, ent, cfg)
        save_json(f"outputs/metrics/{name}.json", {
            "entropy": ent.tolist(),
            "scores": {k: v.tolist() for k, v in scores.items()},
            "types": types,
        })
        print(f"{name}: metrics saved")


if __name__ == "__main__":
    main()