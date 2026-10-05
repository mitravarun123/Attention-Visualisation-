from .base import HFAttentionExtractor


class GPT2Extractor(HFAttentionExtractor):
    name = "gpt2"
    causal = True
    native = True

    def __init__(self, checkpoint="gpt2"):
        super().__init__(checkpoint)