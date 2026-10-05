def clean(tokens):
    """Readable token strings for plots (GPT-2 'Ġ', BERT '##')."""
    out = []
    for t in tokens:
        t = t.replace("Ġ", "").replace("Ċ", "\\n").replace("##", "")
        out.append(t if t else "_")
    return out