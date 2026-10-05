import subprocess
import sys

import pytest
import torch

from attentionlens.extractors import build
from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.head_types import head_scores
from attentionlens.metrics.rollout import attention_rollout
from attentionlens.viz.analysis_figures import head_features
from attentionlens.data.prompts import load_prompts


def test_extractors_import_without_torchscale():
    code = '''
import sys
from attentionlens.extractors.retnet import RetNet
assert not any(n.startswith("torchscale") for n in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_checkpoint_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_retnet.py"):
        build("retnet", {"weights": str(tmp_path / "missing.pt")})


def test_single_token_metrics_are_finite():
    a = torch.ones(2, 3, 1, 1)
    assert torch.equal(head_entropy(a, True), torch.zeros(2, 3))
    assert all(torch.isfinite(v).all() for v in head_scores(a).values())


def test_short_prompt_atlas_features_are_finite():
    for t in (1, 3, 6):
        a = torch.tril(torch.ones(2, 3, t, t))
        assert torch.isfinite(head_features(a, True)).all()


def test_prompt_count():
    assert len(load_prompts()) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_metrics_keep_cuda_device():
    a = torch.tril(torch.ones(2, 3, 8, 8, device="cuda"))
    assert attention_rollout(a).is_cuda
    assert head_entropy(a, True).is_cuda
    assert all(v.is_cuda for v in head_scores(a).values())
    assert head_features(a, True).is_cuda
