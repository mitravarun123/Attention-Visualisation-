from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.head_types import head_scores
from attentionlens.viz.static_figures import TYPE_ORDER, _save

CAUSAL = {"gpt2": True, "bert": False, "retnet": True, "rwkv": True}
MARKERS = ["o", "s", "^", "v", "D", "P", "X"]


def _norm(a):
    p = a.float().abs()
    return p / p.sum(-1, keepdim=True).clamp(min=1e-9)


def _mean_distance(a):
    """Mean |i-j| per query, divided by (T-1). [L, H]."""
    p = _norm(a)
    T = p.shape[-1]
    i = torch.arange(T, device=p.device)
    dist = (i[:, None] - i[None, :]).abs().float()
    return (p * dist).sum((-1, -2)) / T / max(T - 1, 1)


# ---- plot 5: attention vs relative offset ----------------------------------
def offset_profile(data, path, max_d=10):
    """data: {model: [attn [L,H,T,T], ...]}. Mean lookback attention at offset d = i - j."""
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for ci, (name, attns) in enumerate(data.items()):
        vals = []
        for d in range(max_d + 1):
            per = [torch.diagonal(_norm(a), -d, -2, -1).mean().item() for a in attns if a.shape[-1] > d]
            vals.append(np.mean(per) if per else np.nan)
        ax.plot(range(max_d + 1), vals, marker="o", ms=3, color=f"C{ci}", label=name)
    ax.set_yscale("log")
    ax.set_xlabel("relative offset d (query i, key i-d)")
    ax.set_ylabel("mean attention mass")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


# ---- plot 6: mean attention distance, layer x head -------------------------
def distance_heatmap(data, path):
    dist = {n: torch.stack([_mean_distance(a) for a in attns]).mean(0).numpy() for n, attns in data.items()}
    vmax = max(d.max() for d in dist.values())
    fig, axes = plt.subplots(1, len(dist), figsize=(3.3 * len(dist), 3.4), squeeze=False)
    axes = axes[0]
    for ax, (name, d) in zip(axes, dist.items()):
        im = ax.imshow(d, cmap="cividis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="mean |i-j| / (T-1)")
    _save(fig, path)


# ---- plot 7: head atlas -----------------------------------------------------
def head_features(a, causal, K=6):
    """Per-head feature vector [L, H, K+5]."""
    p = _norm(a)
    T = p.shape[-1]
    look = torch.stack([(torch.diagonal(p, -d, -2, -1).mean(-1)
                        if d < T else p.new_zeros(p.shape[:2])) for d in range(K)], -1)
    ahead = torch.triu(p, 1).sum((-1, -2)) / T
    sc = head_scores(a)
    extra = torch.stack([ahead, head_entropy(a, causal), sc["first_token"],
                         sc["local_window"], _mean_distance(a)], -1)
    return torch.cat([look, extra], -1)


def head_atlas(data, types, path):
    """types: {model: [L][H] type names} from outputs/metrics/*.json. PCA of per-head features."""
    feats, models, labels = [], [], []
    for name, attns in data.items():
        f = torch.stack([head_features(a, CAUSAL[name]) for a in attns]).mean(0)   # [L, H, F]
        feats.append(f.reshape(-1, f.shape[-1]))
        models += [name] * (f.shape[0] * f.shape[1])
        labels += [t for row in types[name] for t in row]
    X = torch.cat(feats)
    X = (X - X.mean(0)) / X.std(0, unbiased=False).clamp(min=1e-6)
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    Z = (U[:, :2] * S[:2]).numpy()
    var = ((S ** 2) / (S ** 2).sum().clamp(min=1e-12))[:2].numpy()
    mm, tt = np.array(models), np.array(labels)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for ci, name in enumerate(data):
        for ti, t in enumerate(TYPE_ORDER):
            m = (mm == name) & (tt == t)
            if m.any():
                ax.scatter(Z[m, 0], Z[m, 1], color=f"C{ci}", marker=MARKERS[ti], s=26, alpha=0.8)
    handles = [Line2D([], [], color=f"C{ci}", marker="o", ls="", label=n) for ci, n in enumerate(data)]
    handles += [Line2D([], [], color="gray", marker=MARKERS[ti], ls="", label=t) for ti, t in enumerate(TYPE_ORDER)]
    ax.legend(handles=handles, fontsize=6, ncol=2, frameon=False)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    _save(fig, path)


# ---- plots 8 and 9: layer curves with prompt-variance bands ----------------
def layer_bands(data, fn, ylabel, path):
    """fn(attn, causal) -> [L]. Line = mean over prompts, band = +-1 std over prompts."""
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for ci, (name, attns) in enumerate(data.items()):
        v = torch.stack([fn(a, CAUSAL[name]) for a in attns]).numpy()    # [P, L]
        m, s = v.mean(0), v.std(0)
        x = np.arange(len(m))
        ax.plot(x, m, color=f"C{ci}", marker="o", ms=3, label=name)
        ax.fill_between(x, m - s, m + s, color=f"C{ci}", alpha=0.2)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


def sink_by_layer(data, path):
    layer_bands(data, lambda a, c: _norm(a)[..., 1:, 0].mean((1, 2)), "first-token attention mass", path)


def entropy_bands(data, path):
    layer_bands(data, lambda a, c: head_entropy(a, c).mean(1), "normalised entropy", path)