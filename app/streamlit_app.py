from pathlib import Path

import streamlit as st

from attentionlens.extractors import build
from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.rollout import attention_rollout
from attentionlens.utils.io import load_yaml
from attentionlens.viz.entropy_plot import entropy_by_layer
from attentionlens.viz.heatmaps import attention_map, head_grid, normalised_weights, token_labels
from attentionlens.viz.layer_animation import rollout_animation
from attentionlens.viz.rollout_plot import rollout_heatmap, token_importance

st.set_page_config(page_title="AttentionLens", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem; max-width:1550px;}
[data-testid="stMetric"] {background:#f2f5fa; padding:14px; border-radius:10px;}
</style>""", unsafe_allow_html=True)
st.caption("ATTENTIONLENS / MODEL EXPLORER")
st.title("See what each token attends to")
st.write("Inspect individual heads, follow a token's context, and trace how attention mixes across layers.")
cfg = load_yaml("configs/models.yaml")


@st.cache_resource
def get_extractor(name):
    return build(name, cfg[name])


@st.cache_data
def run(name, text):
    r = get_extractor(name).extract(text)
    return r.attn, r.tokens, r.causal


with st.sidebar:
    st.header("Experiment")
    available = [n for n in cfg if n != "retnet" or Path(cfg[n]["weights"]).is_file()]
    names = st.multiselect("Models", available, default=["gpt2", "bert"])
    if "retnet" not in available:
        st.caption("RetNet unavailable · trained checkpoint required.")
    contrast = st.radio("Colour scale", ["Square root", "Linear"],
                        help="Square root reveals small weights. Hover values remain the actual weights; the colour bar shows actual percentages.")
    st.caption("Rows are queries. Columns are keys. Indexed labels distinguish repeated tokens.")

text = st.text_area("Input text", "The quick brown fox jumps over the lazy dog.", height=100)
if not names or not text.strip():
    st.info("Enter text and select a model to begin.")
    st.stop()
with st.spinner("Extracting attention…"):
    try:
        res = {n: run(n, text) for n in names}
    except (ValueError, OSError, ImportError) as exc:
        st.error(str(exc))
        st.stop()

model = st.selectbox("Inspect model", names)
a, tokens, causal = res[model]
p = normalised_weights(a)
labels = token_labels(tokens)
if model in ("rwkv", "retnet"):
    st.info("This model uses a derived attention analogue. The maps show row-normalised absolute weights; they are not native softmax attention.")
summary = st.columns(4)
summary[0].metric("Tokens", len(tokens))
summary[1].metric("Layers", a.shape[0])
summary[2].metric("Heads" if model not in ("rwkv", "retnet") else "Groups / heads", a.shape[1])
summary[3].metric("Context", "Past + current" if causal else "Bidirectional")
if len(tokens) >= 128:
    st.caption("Input may be truncated at the 128-token extraction limit.")

tabs = st.tabs(["Head explorer", "Compare heads", "Cumulative rollout", "Layer animation", "Model comparison"])
with tabs[0]:
    controls = st.columns(3)
    layer = controls[0].selectbox("Layer", range(a.shape[0]), key="head_layer")
    head = controls[1].selectbox("Head", range(a.shape[1]), key="head_index")
    query = controls[2].selectbox("Query token", range(len(tokens)), index=len(tokens)-1,
                                format_func=lambda i: labels[i], key="head_query")
    st.caption("Read across a row: which context tokens does this query attend to? Orange outlines your selected query. Grey cells are future positions excluded by causality.")
    left, right = st.columns([2.2, 1])
    with left:
        annotate = st.checkbox("Show cell percentages", value=len(tokens)<=16)
        st.plotly_chart(attention_map(p[layer,head], tokens, f"{model.upper()} · Layer {layer} · Head {head}",
                                     causal, contrast, query, annotate), width="stretch", key="detail_map")
    with right:
        row = p[layer,head,query]
        st.markdown(f"**Context for {labels[query]}**")
        st.plotly_chart(token_importance(p[layer,head], tokens, query, "Attention received from selected query"),
                        width="stretch", key="head_bars")
        top = row.argsort(descending=True)[:min(5, query+1 if causal else len(tokens))]
        st.dataframe([{"Key token":labels[i], "Weight":f"{row[i].item():.3%}"} for i in top.tolist()],
                     hide_index=True, width="stretch")
        entropy = head_entropy(p[layer:layer+1,head:head+1], causal).item()
        st.caption(f"Mean normalised head entropy: {entropy:.3f}. Near 0 = concentrated; near 1 = spread across available context.")

with tabs[1]:
    grid_layer = st.selectbox("Comparison layer", range(a.shape[0]), key="grid_layer")
    heads = st.multiselect("Heads to compare", range(a.shape[1]), default=list(range(min(6,a.shape[1]))))
    st.caption("Each panel uses the same colour scale. Compare diagonal, local-context, and first-token patterns across heads.")
    if heads:
        st.plotly_chart(head_grid(a, tokens, grid_layer, heads=heads, causal=causal, contrast=contrast),
                        width="stretch", key="heads_grid")

with tabs[2]:
    through = st.selectbox("Rollout through layer", range(a.shape[0]), index=a.shape[0]-1)
    rq = st.selectbox("Trace query token", range(len(tokens)), index=len(tokens)-1,
                      format_func=lambda i:labels[i], key="rollout_query")
    R = attention_rollout(a[:through+1])
    st.caption("Rollout combines head-averaged weights and a residual connection across layers. It estimates paths to input tokens, rather than one head's direct attention. It is not a causal explanation of model predictions.")
    left, right = st.columns([2.2,1])
    with left:
        st.plotly_chart(rollout_heatmap(R, tokens, f"Rollout · layers 0–{through}", causal, contrast, rq),
                        width="stretch", key="rollout_map")
    with right:
        st.plotly_chart(token_importance(R,tokens,rq,"Paths to original input tokens"), width="stretch", key="rollout_bars")
        st.metric("First-token weight",f"{R[rq,0].item():.2%}")
        st.caption("Small weights can look dark on a linear scale. Square-root contrast reveals them without changing the underlying values.")

with tabs[3]:
    st.caption("Play or select a layer to follow cumulative mixing. All frames share a fixed scale; hover to read actual percentages.")
    st.plotly_chart(rollout_animation(a,tokens,causal,contrast),width="stretch",key="animation")

with tabs[4]:
    st.plotly_chart(entropy_by_layer({n:head_entropy(v[0],v[2]).tolist() for n,v in res.items()}),
                    width="stretch",key="entropy")
    st.caption("Error bars show variation between heads. Each model uses its own tokenisation, so token positions are not directly aligned between models.")
    columns=st.columns(min(2,len(names)))
    for i,n in enumerate(names):
        aa,tt,cc=res[n]
        with columns[i%len(columns)]:
            st.plotly_chart(rollout_heatmap(attention_rollout(aa),tt,f"{n.upper()} · final rollout",cc,contrast),
                            width="stretch",key=f"compare_{n}")
