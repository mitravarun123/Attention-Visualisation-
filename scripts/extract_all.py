import sys
from pathlib import Path

from attentionlens.data.prompts import load_prompts
from attentionlens.extractors import build
from attentionlens.utils.io import load_yaml, save_tensors


def main(models, max_len=128):
    cfg = load_yaml("configs/models.yaml")
    prompts = load_prompts()
    out = Path("outputs/tensors")
    for name in models or list(cfg):
        ext = build(name, cfg[name])
        records = []
        for text in prompts:
            r = ext.extract(text, max_len)
            records.append({"text": text, "tokens": r.tokens, "attn": r.attn.half()})
        save_tensors(out / f"{name}.pt", records)
        print(f"{name}: {len(records)} prompts saved")


if __name__ == "__main__":
    main(sys.argv[1:])