import json
from pathlib import Path
import pytest
from attentionlens.viz.web_analysis import plot, VIEWS, dataset

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not (ROOT/'outputs/tensors/gpt2.pt').exists(), reason='Saved dataset unavailable')
def test_all_interactive_analysis_views_have_finite_serialisable_data():
    for view in VIEWS:
        fig=plot(str(ROOT),view,'gpt2')
        payload=json.loads(fig.to_json())
        assert payload['data'], view
        if view=='head_atlas':
            assert sum(len(t['customdata']) for t in payload['data'])==sum(rr[0]['attn'].shape[0]*rr[0]['attn'].shape[1] for rr in dataset(str(ROOT)).values())
        if view in ('sink_by_layer','entropy_bands'):
            assert sum(t.get('fill')=='toself' for t in payload['data'])==len(dataset(str(ROOT)))
