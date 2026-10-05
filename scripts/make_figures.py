from pathlib import Path

from attentionlens.metrics.rollout import attention_rollout
from attentionlens.utils.io import load_json, load_tensors
from attentionlens.viz import static_figures as sf
from attentionlens.viz.entropy_plot import entropy_by_layer, head_type_bars
from attentionlens.viz.layer_animation import rollout_animation
from attentionlens.viz.side_by_side import compare

OUT = Path("outputs/figures")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {p.stem: load_json(p) for p in Path("outputs/metrics").glob("*.json")}
    first = {p.stem: load_tensors(p)[0] for p in Path("outputs/tensors").glob("*.pt")}

    # interactive (app / GitHub)
    if metrics:
        entropy_by_layer({n: m["entropy"] for n, m in metrics.items()}).write_html(OUT / "entropy_by_layer.html")
        head_type_bars({n: m["types"] for n, m in metrics.items()}).write_html(OUT / "head_types.html")
    if "gpt2" in first and "bert" in first:
        g, b = first["gpt2"], first["bert"]
        compare(attention_rollout(g["attn"].float()), g["tokens"], "GPT-2",
                attention_rollout(b["attn"].float()), b["tokens"], "BERT").write_html(OUT / "gpt2_vs_bert.html")
    for n, rec in first.items():
        rollout_animation(rec["attn"].float(), rec["tokens"], causal=n != "bert", contrast="Square root").write_html(OUT / f"{n}_layer_animation.html")

    # static (paper): PDF + PNG
    if metrics:
        sf.entropy_heatmap({n: m["entropy"] for n, m in metrics.items()}, OUT / "entropy_heatmap")
        sf.head_type_map({n: m["types"] for n, m in metrics.items()}, OUT / "head_type_map")
    for n, rec in first.items():
        sf.layer_snapshots(rec["attn"].float(), rec["tokens"], OUT / f"{n}_layer_snapshots")
    if first:
        sf.all_models_rollout({n: (r["attn"].float(), r["tokens"]) for n, r in first.items()},
                              OUT / "all_models_rollout")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()