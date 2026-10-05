"""Load model-specific dependencies only when an extractor is requested."""
from importlib import import_module

REGISTRY = {
    "gpt2": (".gpt2", "GPT2Extractor"),
    "bert": (".bert", "BERTExtractor"),
    "rwkv": (".rwkv", "RWKVExtractor"),
    "retnet": (".retnet_ts", "TorchScaleRetNetExtractor"),
}


def build(name, cfg=None):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {list(REGISTRY)}")
    options = dict(cfg or {})
    if name == "retnet":
        from pathlib import Path
        weights = options.get("weights")
        if not weights or not Path(weights).is_file():
            raise FileNotFoundError(
                f"RetNet checkpoint missing: {weights}. Provide benchmark weights or "
                "run python scripts/train_retnet.py to train the configured TorchScale model."
            )
    module, cls = REGISTRY[name]
    try:
        extractor = getattr(import_module(module, __name__), cls)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "torchscale":
            raise ImportError("RetNet requires: pip install -e '.[retnet]'") from exc
        raise
    return extractor(**options)
