from .base import HFAttentionExtractor


class BERTExtractor(HFAttentionExtractor):
    name = "bert"
    causal = False
    native = True

    def __init__(self, checkpoint="bert-base-uncased"):
        super().__init__(checkpoint)