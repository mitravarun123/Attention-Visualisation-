import plotly.graph_objects as go
from attentionlens.viz.heatmaps import attention_map, token_labels


def rollout_heatmap(R, tokens, title="Attention rollout", causal=False, contrast="Linear", query=None):
    return attention_map(R, tokens, title, causal, contrast, query)


def token_importance(R, tokens, query=-1, title="Token importance"):
    labels = token_labels(tokens)
    fig = go.Figure(go.Bar(x=R[query].detach().float().cpu().numpy(), y=labels,
                          orientation="h", marker_color="#2563a6",
                          hovertemplate="%{y}<br>Weight: %{x:.4%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", title=title, xaxis_title="Weight", xaxis_tickformat=".0%",
                      height=max(360, len(tokens)*26+100), margin=dict(l=100,t=60,b=40))
    return fig
