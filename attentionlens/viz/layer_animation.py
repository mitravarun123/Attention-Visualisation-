import plotly.graph_objects as go
from attentionlens.metrics.rollout import attention_rollout
from attentionlens.viz.heatmaps import matrix_trace, token_labels


def rollout_animation(attn, tokens, causal=False, contrast="Linear"):
    cum = attention_rollout(attn, cumulative=True)
    frames = [go.Frame(data=[matrix_trace(cum[l], tokens, causal, contrast)], name=str(l),
                       layout=go.Layout(title=f"Cumulative rollout · through layer {l}"))
              for l in range(cum.shape[0])]
    fig = go.Figure(data=frames[0].data, frames=frames)
    labels = token_labels(tokens)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(tokens))), ticktext=labels, tickangle=-55,
                     title="Key · original input token")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(tokens))), ticktext=labels,
                     autorange="reversed", title="Query · representation token")
    options = {"frame": {"duration": 650, "redraw": True}, "transition": {"duration": 0}}
    fig.update_layout(template="plotly_white", height=650, margin=dict(l=120,b=170),
        title="Cumulative rollout · through layer 0", plot_bgcolor="#edf0f4",
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="▶ Play", method="animate", args=[None, options]),
            dict(label="Pause", method="animate", args=[[None], {"mode":"immediate", "frame":{"duration":0,"redraw":False}}])])],
        sliders=[dict(currentvalue=dict(prefix="Through layer "), steps=[
            dict(label=str(l), method="animate", args=[[str(l)],
                {"mode":"immediate", "frame":{"duration":0,"redraw":True}, "transition":{"duration":0}}])
            for l in range(cum.shape[0])])])
    return fig
