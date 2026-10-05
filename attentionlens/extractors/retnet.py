import math
import warnings

import torch
import torch.nn as nn

from attentionlens.utils.tokens import clean
from .base import AttentionExtractor, AttentionResult


class Retention(nn.Module):
    """Multi-scale retention, parallel form: (QK^T / sqrt(d) * D) V. No xpos/rotary."""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.register_buffer("gamma", 1 - 2.0 ** (-5 - torch.arange(h).float()))

    def forward(self, x):
        B, T, D = x.shape
        sp = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        i = torch.arange(T, device=x.device)
        diff = i[:, None] - i[None, :]
        decay = (self.gamma[:, None, None] ** diff.clamp(min=0)) * (diff >= 0)
        R = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) * decay   # [B, H, T, T]
        R = R / R.abs().sum(-1, keepdim=True).detach().clamp(min=1)
        out = (R @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out), R


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ret = Retention(d, h)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        a, R = self.ret(self.ln1(x))
        x = x + a
        return x + self.ffn(self.ln2(x)), R


class RetNet(nn.Module):
    def __init__(self, vocab, d=768, layers=12, heads=12):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, ids):
        """Returns retention matrices [L, B, H, T, T]."""
        x, Rs = self.emb(ids), []
        for b in self.blocks:
            x, R = b(x)
            Rs.append(R)
        return torch.stack(Rs)


class RetNetExtractor(AttentionExtractor):
    name = "retnet"
    causal = True
    native = False

    def __init__(self, tokenizer="gpt2", weights=None, d_model=768, layers=12, heads=12):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.model = RetNet(len(self.tok), d_model, layers, heads).eval()
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        else:
            warnings.warn("RetNet has random weights; patterns are not meaningful until you load trained ones.")

    @torch.no_grad()
    def extract(self, text, max_len=128):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        attn = self.model(enc["input_ids"])[:, 0].float()  # [L, H, T, T]
        tokens = clean(self.tok.convert_ids_to_tokens(enc["input_ids"][0]))
        return AttentionResult(attn, tokens, self.name, self.causal, self.native)