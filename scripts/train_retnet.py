from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from attentionlens.extractors.retnet_ts import RetNetModel
from attentionlens.utils.io import load_yaml


def main(token_budget=20_000_000, seq=256, bs=16, lr=3e-4, out=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_yaml("configs/models.yaml")["retnet"]
    out = out or cfg["weights"]
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "gpt2"))
    if len(tok) != cfg["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match RetNet configuration")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    chunks, n = [], 0
    for ex in ds:
        if ex["text"].strip():
            ids = np.array(tok(ex["text"])["input_ids"], dtype=np.int32)
            chunks.append(ids)
            n += len(ids)
        if n >= token_budget:
            break
    if not chunks:
        raise ValueError("Dataset produced no tokens")
    flat = np.concatenate(chunks)
    data = torch.from_numpy(flat[: len(flat) // seq * seq].astype(np.int64)).view(-1, seq)

    if len(data) < bs:
        raise ValueError("Token budget must provide at least one full batch")
    model = RetNetModel(cfg["vocab_size"], cfg["embed_dim"],
                        cfg["num_heads"], cfg["num_layers"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    steps = len(data) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    use_amp = dev == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    perm = torch.randperm(len(data))

    for s in range(steps):
        batch = data[perm[s * bs:(s + 1) * bs]].to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if s % 100 == 0:
            print(f"step {s}/{steps}  loss {loss.item():.3f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("saved", out, "| final loss", loss.item())


if __name__ == "__main__":
    main()