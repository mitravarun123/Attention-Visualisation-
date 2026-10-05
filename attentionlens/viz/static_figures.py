from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from attentionlens.metrics.rollout import attention_rollout

TYPE_ORDER = ["first_token", "previous_token", "next_token", "self", "local_window", "broad", "other"]
TYPE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#cccccc"]


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ticks(ax, tokens, fs=5):
    if len(tokens) <= 30:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=fs)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=fs)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def layer_snapshots(attn, tokens, path, layers=None):
    """Static replacement for the layer animation: cumulative rollout at chosen layers."""
    cum = attention_rollout(attn, cumulative=True).numpy()
    L = cum.shape[0]
    layers = layers or sorted({0, L // 4, L // 2, 3 * L // 4, L - 1})
    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3.4), squeeze=False)
    axes = axes[0]
    for ax, l in zip(axes, layers):
        im = ax.imshow(cum[l], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Layer {l}", fontsize=9)
        _ticks(ax, tokens)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="cumulative rollout")
    _save(fig, path)


def entropy_heatmap(results, path):
    """results: {model: [L][H] normalised entropy}. One layer x head heatmap per model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    axes = axes[0]
    for ax, (name, ent) in zip(axes, results.items()):
        im = ax.imshow(np.array(ent), cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="normalised entropy")
    _save(fig, path)


def head_type_map(types_by_model, path):
    """types_by_model: {model: [L][H] type names}. Layer x head grid coloured by head type."""
    n = len(types_by_model)
    cmap = ListedColormap(TYPE_COLORS)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    axes = axes[0]
    for ax, (name, types) in zip(axes, types_by_model.items()):
        z = np.array([[TYPE_ORDER.index(t) for t in row] for row in types])
        ax.imshow(z, cmap=cmap, vmin=0, vmax=len(TYPE_ORDER) - 1, aspect="auto")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
    fig.legend(handles=[Patch(color=c, label=t) for t, c in zip(TYPE_ORDER, TYPE_COLORS)],
               loc="lower center", ncol=len(TYPE_ORDER), fontsize=7, frameon=False)
    fig.subplots_adjust(bottom=0.2)
    _save(fig, path)


def all_models_rollout(models, path):
    """models: {name: (attn [L,H,T,T], tokens)}. Final rollout for every model side by side."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    axes = axes[0]
    for ax, (name, (attn, tokens)) in zip(axes, models.items()):
        im = ax.imshow(attention_rollout(attn).numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        _ticks(ax, tokens)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="rollout")
    _save(fig, path)