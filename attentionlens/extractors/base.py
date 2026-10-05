from dataclasses import dataclass

import torch

from attentionlens.utils.tokens import clean


@dataclass
class AttentionResult:
    attn: torch.Tensor      # [L, H, T, T]
    tokens: list
    model_name: str
    causal: bool
    native: bool            # True = real softmax attention, False = derived analogue


class AttentionExtractor:
    name = "base"
    causal = True
    native = True

    def extract(self, text, max_len=128) -> AttentionResult:
        raise NotImplementedError


class HFAttentionExtractor(AttentionExtractor):
    def __init__(self, checkpoint):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint, attn_implementation="eager").eval()

    @torch.no_grad()
    def extract(self, text, max_len=128):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        out = self.model(**enc, output_attentions=True)
        attn = torch.stack(out.attentions)[:, 0].float()  # [L, H, T, T]
        tokens = clean(self.tok.convert_ids_to_tokens(enc["input_ids"][0]))
        return AttentionResult(attn, tokens, self.name, self.causal, self.native)