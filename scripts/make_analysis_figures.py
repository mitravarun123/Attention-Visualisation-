from pathlib import Path

from attentionlens.utils.io import load_json, load_tensors
from attentionlens.viz import analysis_figures as af

OUT = Path("outputs/figures")


def main():
    data = {p.stem: [r["attn"].float() for r in load_tensors(p)]
            for p in sorted(Path("outputs/tensors").glob("*.pt"))}
    data = {name: attns for name, attns in data.items() if attns}
    if not data:
        raise ValueError("No extracted tensors found; run scripts/extract_all.py first")
    types = {name: load_json(f"outputs/metrics/{name}.json")["types"] for name in data}
    OUT.mkdir(parents=True, exist_ok=True)
    af.offset_profile(data, OUT / "offset_profile")
    af.distance_heatmap(data, OUT / "distance_heatmap")
    af.head_atlas(data, types, OUT / "head_atlas")
    af.sink_by_layer(data, OUT / "sink_by_layer")
    af.entropy_bands(data, OUT / "entropy_bands")
    print("analysis figures written to", OUT)


if __name__ == "__main__":
    main()
