import torch
import torch.nn as nn
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

from attentionlens.utils.tokens import clean
from .base import AttentionExtractor, AttentionResult


class RetNetModel(nn.Module):
    """Same structure and parameter names as the benchmark class, so its state_dict loads."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.1, value_embed_dim=1280):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        config = RetNetConfig(
            decoder_embed_dim=embed_dim,
            decoder_value_embed_dim=value_embed_dim,
            decoder_retention_heads=num_heads,
            decoder_ffn_embed_dim=embed_dim * 4,
            decoder_layers=num_layers,
            dropout=dropout,
            activation_dropout=dropout,
            no_scale_embedding=True,
            layernorm_embedding=False,
            vocab_size=vocab_size,
        )
        self.retnet = RetNetDecoder(config, embed_tokens=self.token_emb)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        out, _ = self.retnet(prev_output_tokens=x, features_only=True)
        return self.fc_out(out)


class TorchScaleRetNetExtractor(AttentionExtractor):
    name = "retnet"
    causal = True
    native = False

    def __init__(self, weights, vocab_size, tokenizer="gpt2", embed_dim=768, num_heads=12, num_layers=12):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        value_dim = state["retnet.layers.0.retention.v_proj.weight"].shape[0]
        self.model = RetNetModel(vocab_size, embed_dim, num_heads, num_layers,
                                 value_embed_dim=value_dim)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.num_layers = num_layers
        self._store = []
        for layer in self.model.retnet.layers:
            self._patch(layer.retention)

    def _patch(self, ret):
        orig = ret.parallel_forward

        def patched(qr, kr, v, mask):
            qk = (qr @ kr.transpose(-1, -2)) * mask
            qk = qk / qk.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
            self._store.append(qk.detach())          # [B, H, T, T]
            return orig(qr, kr, v, mask)

        ret.parallel_forward = patched

    @torch.no_grad()
    def extract(self, text, max_len=128):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        self._store.clear()
        self.model(enc["input_ids"])
        assert len(self._store) == self.num_layers, "parallel_forward was not called once per layer"
        attn = torch.stack(self._store)[:, 0].float()  # [L, H, T, T]
        tokens = clean(self.tok.convert_ids_to_tokens(enc["input_ids"][0]))
        return AttentionResult(attn, tokens, self.name, self.causal, self.native)