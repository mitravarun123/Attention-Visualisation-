import plotly.graph_objects as go

TYPE_ORDER = ["first_token", "previous_token", "next_token", "self", "local_window", "broad", "other"]


def entropy_by_layer(results):
    """results: {model_name: [L][H] normalised entropies}. Mean per layer with std band."""
    import numpy as np
    fig = go.Figure()
    for name, ent in results.items():
        e = np.array(ent)
        m, s = e.mean(1), e.std(1)
        x = list(range(len(m)))
        fig.add_trace(go.Scatter(x=x, y=m, mode="lines+markers", name=name,
                                 error_y=dict(type="data", array=s, visible=True)))
    fig.update_layout(title="Normalised head entropy by layer", xaxis_title="layer", yaxis_title="entropy")
    return fig


def head_type_bars(types_by_model):
    """types_by_model: {model_name: [L][H] type names}. Fraction of heads per type."""
    fig = go.Figure()
    for t in TYPE_ORDER:
        ys = []
        for name, types in types_by_model.items():
            flat = [x for row in types for x in row]
            ys.append(flat.count(t) / len(flat))
        fig.add_trace(go.Bar(name=t, x=list(types_by_model), y=ys))
    fig.update_layout(barmode="stack", title="Head type distribution", yaxis_title="fraction of heads")
    return fig