"""Local React API. Run with uvicorn app.api:app."""
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from attentionlens.extractors import build
from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.rollout import attention_rollout
from attentionlens.utils.io import load_yaml, load_tensors

ROOT = Path(__file__).resolve().parents[1]
CFG = load_yaml(ROOT / 'configs/models.yaml')
app = FastAPI(title='AttentionLens')
lock = Lock()


class Extraction(BaseModel):
    model: str
    text: str = Field(min_length=1, max_length=12000)


@app.get('/api/models')
def models():
    return [{"id": n, "available": n != 'retnet' or (ROOT / c['weights']).is_file(),
             "native": n in ('gpt2', 'bert'), "causal": n != 'bert'} for n,c in CFG.items()]


@lru_cache(maxsize=4)
def extractor(name):
    cfg = dict(CFG[name])
    if 'weights' in cfg:
        cfg['weights'] = str(ROOT / cfg['weights'])
    return build(name, cfg)


def serialise(a, tokens, name):
    p = a.float().abs()
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
    return {"model": name, "tokens": tokens, "causal": name != 'bert',
            "native": name in ('gpt2', 'bert'), "attention": p.tolist(),
            "rollout": attention_rollout(a, cumulative=True).tolist(),
            "entropy": head_entropy(a, name != 'bert').tolist()}


@lru_cache(maxsize=12)
def extract(name, text):
    # Reuse exact saved prompts when available. New text runs the real model.
    path = ROOT / f'outputs/tensors/{name}.pt'
    if path.exists():
        for rec in load_tensors(path):
            if rec['text'] == text:
                return serialise(rec['attn'], rec['tokens'], name)
    with lock:  # RWKV hooks and RetNet stores are mutable per extractor.
        result = extractor(name).extract(text)
        return serialise(result.attn, result.tokens, name)


@app.post('/api/extract')
def extract_route(request: Extraction):
    if request.model not in CFG:
        raise HTTPException(400, 'Unknown model')
    if not request.text.strip():
        raise HTTPException(400, 'Enter some text')
    if request.model == 'retnet' and not (ROOT / CFG['retnet']['weights']).is_file():
        raise HTTPException(409, 'RetNet requires a trained checkpoint')
    try:
        return {**extract(request.model, request.text), "text": request.text}
    except (OSError, ValueError, ImportError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get('/api/health')
def health():
    return {"status":"ok"}


DIST = ROOT / 'frontend/dist'
if DIST.exists():
    app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')


@app.get('/')
def index():
    if not (DIST / 'index.html').is_file():
        raise HTTPException(503, 'Build frontend first')
    return FileResponse(DIST / 'index.html')


@app.get('/api/analysis')
def analysis_metadata():
    from attentionlens.viz.web_analysis import dataset, VIEWS
    records = dataset(str(ROOT))
    return {"views": [{"id": k, "title": v[0], "description": v[1]} for k,v in VIEWS.items()],
            "models": [{"id": n, "prompts": len(rr)} for n,rr in records.items()],
            "missing": [n for n in CFG if n not in records]}


@app.get('/api/plotly.js')
def plotly_script():
    from plotly.offline import get_plotlyjs
    return Response(get_plotlyjs(), media_type='application/javascript', headers={"Cache-Control":"public, max-age=86400"})


@app.get('/api/analysis/{view}', response_class=HTMLResponse)
def analysis_figure(view: str, model: str = 'gpt2', dark: bool = False):
    from attentionlens.viz.web_analysis import plot
    try:
        figure = plot(str(ROOT), view, model, dark)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    html = figure.to_html(include_plotlyjs='/api/plotly.js', full_html=True,
                          config={"responsive":True, "displaylogo":False, "toImageButtonOptions":{"format":"png","scale":2}})
    background = '#182233' if dark else '#ffffff'
    return html.replace('<body>', f'<body style="margin:0;background:{background}">')
