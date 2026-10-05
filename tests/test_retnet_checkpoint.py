import pytest
import torch
pytest.importorskip('torchscale')
from attentionlens.extractors.retnet_ts import RetNetModel


def test_retnet_supports_checkpoint_value_projection_width():
    source=RetNetModel(100,32,4,2,dropout=0,value_embed_dim=32).eval()
    target=RetNetModel(100,32,4,2,dropout=0,value_embed_dim=32).eval()
    target.load_state_dict(source.state_dict(),strict=True)
    ids=torch.randint(0,100,(1,8))
    assert source.retnet.layers[0].retention.v_proj.weight.shape==(32,32)
    with torch.no_grad():
        assert torch.allclose(source(ids),target(ids))
