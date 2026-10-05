"""Readable attention maps with positional labels and faithful hover values."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def token_labels(tokens):
    return [f"{i:02d} · {token}" for i, token in enumerate(tokens)]


def normalised_weights(attn):
    p = attn.detach().float().abs()
    return p / p.sum(-1, keepdim=True).clamp(min=1e-9)


def matrix_trace(matrix, tokens, causal=False, contrast="Linear", query=None,
                 annotations=False, showscale=True):
    raw = matrix.detach().float().cpu().numpy().copy()
    visible = raw.copy()
    if causal:
        visible[np.triu_indices(len(tokens), 1)] = np.nan
    enhanced = contrast == "Square root"
    z = np.sqrt(visible) if enhanced else visible
    ticks = [0, .01, .05, .1, .25, .5, 1]
    colorbar = dict(title="Weight", tickvals=np.sqrt(ticks) if enhanced else ticks,
                    ticktext=[f"{v:.0%}" for v in ticks], thickness=14)
    labels = token_labels(tokens)
    custom = np.empty((*raw.shape, 3), dtype=object)
    for i, q in enumerate(labels):
        for j, k in enumerate(labels):
            custom[i, j] = [q, k, raw[i, j]]
    text = [[f"{v:.1%}" if np.isfinite(v) else "" for v in row] for row in visible]
    return go.Heatmap(z=z, x=list(range(len(tokens))), y=list(range(len(tokens))),
                      customdata=custom, colorscale="Blues", zmin=0, zmax=1,
                      colorbar=colorbar, showscale=showscale, xgap=1, ygap=1,
                      text=text, texttemplate="%{text}" if annotations else None,
                      hovertemplate="Query: %{customdata[0]}<br>Key: %{customdata[1]}"
                                    "<br>Weight: %{customdata[2]:.4%}<extra></extra>",
                      hoverongaps=False)


def attention_map(matrix, tokens, title="Attention", causal=False, contrast="Linear",
                  query=None, annotations=False):
    fig = go.Figure(matrix_trace(matrix, tokens, causal, contrast, query, annotations))
    labels = token_labels(tokens)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(tokens))), ticktext=labels,
                     tickangle=-55, title="Key · token receiving attention", side="bottom")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(tokens))), ticktext=labels,
                     title="Query · token looking at context", autorange="reversed")
    fig.update_layout(template="plotly_white", title=title, height=max(520, min(1000, len(tokens)*27+200)),
                      margin=dict(l=120, r=40, t=60, b=150), plot_bgcolor="#edf0f4")
    if query is not None:
        fig.add_shape(type="rect", x0=-.5, x1=len(tokens)-.5, y0=query-.5, y1=query+.5,
                      line=dict(color="#e58924", width=2), fillcolor="rgba(0,0,0,0)")
    return fig


def head_grid(attn, tokens, layer, cols=3, heads=None, causal=False, contrast="Linear"):
    heads = list(range(attn.shape[1])) if heads is None else list(heads)
    if not heads:
        raise ValueError("Select at least one head")
    cols = min(cols, len(heads))
    rows = -(-len(heads) // cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Layer {layer} · Head {h}" for h in heads],
                        horizontal_spacing=.10, vertical_spacing=min(.12, .5/rows))
    p = normalised_weights(attn)
    labels = token_labels(tokens)
    for position, h in enumerate(heads):
        row, col = position // cols + 1, position % cols + 1
        fig.add_trace(matrix_trace(p[layer,h], tokens, causal, contrast, showscale=False), row=row, col=col)
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(tokens))), ticktext=labels,
                         tickangle=-60, tickfont=dict(size=9), row=row, col=col)
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(tokens))), ticktext=labels,
                         tickfont=dict(size=9), autorange="reversed", row=row, col=col)
    fig.update_layout(template="plotly_white", height=440*rows, margin=dict(l=80,r=30,t=65,b=100),
                      title=f"Head comparison · Layer {layer}", plot_bgcolor="#edf0f4")
    return fig
