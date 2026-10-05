import numpy as np
import torch
from attentionlens.viz.heatmaps import attention_map
from attentionlens.viz.layer_animation import rollout_animation


def test_repeated_tokens_use_distinct_positions_and_actual_hover_weights():
    a=torch.tensor([[1.,0.],[.01,.99]])
    fig=attention_map(a,['echo','echo'],causal=True,contrast='Square root')
    trace=fig.data[0]
    assert list(trace.x)==[0,1]
    assert fig.layout.xaxis.ticktext[0]!=fig.layout.xaxis.ticktext[1]
    assert np.isnan(trace.z[0,1])
    assert np.isclose(trace.z[1,0],.1)
    assert np.isclose(trace.customdata[1,0,2],.01)


def test_animation_preserves_scale_and_causal_mask_across_frames():
    a=torch.tensor([[1.,0.],[.25,.75]]).expand(3,2,2,2)
    fig=rollout_animation(a,['a','b'],causal=True,contrast='Square root')
    assert len(fig.frames)==3
    for frame in fig.frames:
        trace=frame.data[0]
        assert trace.zmin==0 and trace.zmax==1
        assert np.isnan(trace.z[0,1])
