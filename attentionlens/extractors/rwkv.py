import torch

from attentionlens.utils.tokens import clean
from .base import AttentionExtractor, AttentionResult


class RWKVExtractor(AttentionExtractor):
    """Implicit attention from the RWKV-4 WKV recurrence.

    Per channel: weight(t, i<t) ~ exp((t-1-i) * w + k_i), weight(t, t) ~ exp(u + k_t),
    softmax over i. Channels are averaged within n_groups groups to give pseudo-heads.
    """
    name = "rwkv"
    causal = True
    native = False

    def __init__(self, checkpoint="RWKV/rwkv-4-169m-pile", n_groups=12):
        from transformers import AutoTokenizer, RwkvModel
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = RwkvModel.from_pretrained(checkpoint).eval()
        self.n_groups = n_groups

    @staticmethod
    def _implicit(k, time_decay, time_first, groups):
        T, C = k.shape
        w = -torch.exp(time_decay)                          # [C], negative
        i = torch.arange(T)
        dist = (i[:, None] - 1 - i[None, :]).float()        # t-1-i
        logits = dist[:, :, None] * w[None, None, :] + k[None, :, :]   # [T, T, C]
        logits[i, i] = time_first + k                       # current token term
        logits = logits.masked_fill((i[None, :] > i[:, None])[:, :, None], float("-inf"))
        attn = torch.softmax(logits, dim=1)
        attn = attn.view(T, T, groups, C // groups).mean(-1)
        return attn.permute(2, 0, 1)                        # [G, T, T]

    @torch.no_grad()
    def extract(self, text, max_len=128):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        keys = []
        hooks = [b.attention.key.register_forward_hook(
            lambda m, i, o: keys.append(o.detach()[0].float())) for b in self.model.blocks]
        try:
            self.model(input_ids=enc["input_ids"], use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        mats = [self._implicit(k, b.attention.time_decay.float(), b.attention.time_first.float(), self.n_groups)
                for b, k in zip(self.model.blocks, keys)]
        tokens = clean(self.tok.convert_ids_to_tokens(enc["input_ids"][0]))
        return AttentionResult(torch.stack(mats), tokens, self.name, self.causal, self.native)