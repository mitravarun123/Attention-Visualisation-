import torch

from attentionlens.metrics.rollout import attention_rollout


def test_identity_attention_gives_identity_rollout():
    attn = torch.eye(6).expand(3, 2, 6, 6)
    assert torch.allclose(attention_rollout(attn), torch.eye(6), atol=1e-5)


def test_rows_sum_to_one():
    attn = torch.softmax(torch.randn(4, 3, 8, 8), -1)
    assert torch.allclose(attention_rollout(attn).sum(-1), torch.ones(8), atol=1e-4)