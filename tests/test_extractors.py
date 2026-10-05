import torch

from attentionlens.extractors.retnet import RetNet


def test_retnet_shapes_and_causality():
    model = RetNet(vocab=100, d=32, layers=2, heads=4).eval()
    R = model(torch.randint(0, 100, (1, 10)))[:, 0]   # [L, H, T, T]
    assert R.shape == (2, 4, 10, 10)
    assert torch.triu(R, diagonal=1).abs().sum() == 0