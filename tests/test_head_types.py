import torch

from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.head_types import classify, head_scores
from attentionlens.utils.io import load_yaml

CFG = load_yaml("configs/classifier.yaml")
T = 20


def _label(p, causal=True):
    attn = p[None, None]
    return classify(head_scores(attn, CFG["window"]), head_entropy(attn, causal), CFG)[0][0]


def test_previous_token():
    p = torch.zeros(T, T)
    p[0, 0] = 1
    for i in range(1, T):
        p[i, i - 1] = 1
    assert _label(p) == "previous_token"


def test_first_token():
    p = torch.zeros(T, T)
    p[:, 0] = 1
    assert _label(p) == "first_token"


def test_local_window():
    p = torch.zeros(T, T)
    for i in range(T):
        p[i, max(0, i - 3): i + 1] = 1
    p = p / p.sum(-1, keepdim=True)
    assert _label(p) == "local_window"


def test_uniform_causal_has_entropy_one():
    p = torch.tril(torch.ones(T, T))
    p = p / p.sum(-1, keepdim=True)
    assert abs(head_entropy(p[None, None], True).item() - 1.0) < 1e-3