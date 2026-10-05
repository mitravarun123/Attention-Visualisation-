# Paper figures 1–6

All aggregate comparisons use the same 32 prompts, with equal prompt weights. Every figure has a vector PDF, 400-dpi PNG, and CSV source table.

1. Final rollout: illustrative first saved prompt, indexed model-specific tokens. Selected query is the final text token, excluding BERT's [SEP]. Only that row is numerically annotated. Shared square-root colour mapping; percentages refer to original rollout weights. Rollout averages heads, adds a residual identity connection and multiplies layer transitions.
2. Entropy: per-row entropy divided by log of available context size; first causal row excluded. Head mean per layer, then prompt mean ±1 population SD. Final-layer table gives mean and SD.
3. Head types: classify prompt-averaged mass scores and entropy with configs/classifier.yaml. Categories are mutually exclusive through priority ordering. Table includes every category's percentage and count; RWKV groups are pseudo-heads.
4. Relative distance: expected absolute token offset divided by T−1, mean over queries and prompts. Shared colour scale; every cell annotated in percent. Model means additionally average layers and heads.
5. First-token mass: mean over all heads and queries except query zero, then prompt mean ±1 population SD. The first token is BERT's [CLS], versus the first text token for causal models; these are not identical semantic positions.
6. Lookback profile: mean diagonal weight over layers, heads, and eligible queries in each prompt, then mean across eligible prompts. Logarithmic y-axis; table reports exact percentages to two decimal places and eligible prompt counts. Forward-looking BERT weights are outside this figure.

## Interpretation and provenance
GPT-2 and BERT use native softmax attention. RetNet uses signed retention magnitudes, and RWKV uses channel-grouped implicit weights; these analogues are row-normalised for metrics. Training corpora, objectives, tokenizer boundaries, special tokens, and checkpoint maturity differ. These are descriptive pattern comparisons, not controlled evidence of architectural superiority or prediction causality.

RetNet: user-provided TinyStories-trained checkpoint; user-reported validation loss 4.434524583816528. training_metadata.json now records 100 optimizer steps, sequence length 128, batch size 1 and accumulation 8; validation loss is recorded there. Exact GPU and validation sample size are not recorded. The checkpoint loads strictly with 768-dimensional value projections. All input tensors were saved in half precision by the extraction pipeline, then converted to float32 for figure calculations.
