from plotly.subplots import make_subplots
from attentionlens.viz.heatmaps import matrix_trace, token_labels


def compare(R_a, tokens_a, name_a, R_b, tokens_b, name_b):
    """Each model retains its own indexed tokenisation; colour scales match."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=[name_a, name_b], horizontal_spacing=.15)
    for col, (R, tokens, name) in enumerate([(R_a,tokens_a,name_a),(R_b,tokens_b,name_b)],1):
        fig.add_trace(matrix_trace(R,tokens,name.lower() in ('gpt2','rwkv','retnet'),
                                  "Square root",showscale=col==2),row=1,col=col)
        labels=token_labels(tokens)
        fig.update_xaxes(tickmode="array",tickvals=list(range(len(tokens))),ticktext=labels,
                         tickangle=-60,row=1,col=col)
        fig.update_yaxes(tickmode="array",tickvals=list(range(len(tokens))),ticktext=labels,
                         autorange="reversed",row=1,col=col)
    fig.update_layout(template="plotly_white",title=f"{name_a} vs {name_b}: attention rollout",
                      height=650,margin=dict(b=160),plot_bgcolor="#edf0f4")
    return fig
