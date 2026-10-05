# AttentionLens
Cross-architecture attention visualisation: GPT-2, BERT, RetNet, RWKV.

    pip install -e .
    python scripts/extract_all.py gpt2 bert rwkv retnet
    python scripts/compute_metrics.py
    python scripts/make_figures.py
    streamlit run app/streamlit_app.py
    pytest
RetNet uses the TorchScale architecture for both training and extraction.
Install its optional dependencies with `pip install -e '.[retnet]'`, or
`pip install -e '.[training]'` to train. Place matching benchmark weights at
`checkpoints/retnet_128.pt`, or run `python scripts/train_retnet.py` to train
and save the model at the configured path. Training downloads WikiText and
can require substantial compute. GPT-2, BERT, RWKV, and the core tests do not
require TorchScale.

Run `python scripts/make_analysis_figures.py` after extraction and metrics to
produce the five analysis figures as PDF and PNG. Run
`python scripts/sensitivity.py` to compare classifier windows and thresholds;
results are saved under `outputs/metrics/sensitivity/`.

Local virtual environment:

```sh
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python scripts/extract_all.py gpt2 bert rwkv
python scripts/compute_metrics.py
python scripts/make_figures.py
python scripts/make_analysis_figures.py
python scripts/sensitivity.py
streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Once models are downloaded, set `HF_HUB_OFFLINE=1` to reuse the cache without
network requests. Add `retnet` to extraction after providing the checkpoint.

The interactive explorer offers indexed token labels, a selected-query row and
context ranking, selected-head comparisons, and rollout through a chosen layer.
Square-root colour contrast reveals small weights; tooltips and colour-bar labels
always show the original weights. Grey future cells mark causal masking. RWKV
and RetNet maps use normalised absolute analogue weights, not softmax attention.

## React interface + Python API

The primary interface is now React, served by FastAPI on the same local origin.
The original Streamlit app remains available separately.

```sh
source .venv/bin/activate
pip install -e '.[web]'
cd frontend
npm install
npm run build
cd ..
python scripts/run_web.py
```

Open http://127.0.0.1:8501. Stop any Streamlit process on port 8501 first.
Saved prompts reuse extracted tensors; new inputs use the actual models. Model
files use `.cache/huggingface`. Set `HF_HUB_OFFLINE=1` once the required models
are downloaded to run without network access. RetNet becomes available when
its configured trained checkpoint is present; restart the server after adding it.
The frontend includes indexed-token attention maps, linked query rankings,
head comparisons, entropy, rollout playback, JSON export, and light/dark themes.
Rebuild the frontend and restart the API after source edits.

The **Dataset analysis** workspace includes final rollout comparison, cumulative
layer snapshots, head-type maps, lookback offset profiles, relative attention
distance, first-token mass by layer, the PCA head atlas, and entropy curves with
prompt-variance bands. Dataset trends average saved prompts equally. Bands are
±1 population standard deviation across prompts, not confidence intervals.
Rollout panels use the first saved prompt, labelled in the chart. Charts offer
hover values, zoom, trace toggles, and PNG export. Analysis is cached until the
server restarts; restart after regenerating saved tensors.
# Attention-Visualisation-
