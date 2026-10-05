"""Reclassify saved attention under different windows and thresholds."""
import argparse
from pathlib import Path
from collections import Counter

import torch

from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.head_types import classify, head_scores
from attentionlens.utils.io import load_tensors, load_yaml, save_json

CAUSAL = {"gpt2": True, "bert": False, "rwkv": True, "retnet": True}


def main(windows=(1, 3, 5), deltas=(-0.1, 0.0, 0.1)):
    cfg = load_yaml("configs/classifier.yaml")
    results = {}
    for path in sorted(Path("outputs/tensors").glob("*.pt")):
        attns = [r["attn"].float() for r in load_tensors(path)]
        if not attns:
            continue
        entropy = torch.stack([head_entropy(a, CAUSAL[path.stem]) for a in attns]).mean(0)
        runs = []
        for window in windows:
            per = [head_scores(a, window) for a in attns]
            scores = {k: torch.stack([s[k] for s in per]).mean(0) for k in per[0]}
            for delta in deltas:
                thresholds = {k: min(1.0, max(0.0, v + delta))
                              for k, v in cfg.items() if k != "window"}
                types = classify(scores, entropy, thresholds)
                counts = Counter(t for row in types for t in row)
                runs.append({"window": window, "threshold_delta": delta,
                             "counts": dict(counts), "types": types})
        results[path.stem] = runs
    if not results:
        raise ValueError("No extracted tensors found; run scripts/extract_all.py first")
    save_json("outputs/metrics/sensitivity/results.json", results)
    print("saved outputs/metrics/sensitivity/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--deltas", nargs="+", type=float, default=[-0.1, 0.0, 0.1])
    args = parser.parse_args()
    if any(w < 0 for w in args.windows):
        parser.error("windows must be nonnegative")
    main(args.windows, args.deltas)
