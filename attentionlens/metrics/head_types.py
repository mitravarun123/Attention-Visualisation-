import torch

TYPES = ["first_token", "previous_token", "next_token", "self", "local_window", "broad", "other"]


def _norm(attn):
    p = attn.float().abs()
    return p / p.sum(-1, keepdim=True).clamp(min=1e-9)


def head_scores(attn, window=3):
    """attn: [L, H, T, T] -> dict of [L, H] mean attention-mass scores."""
    p = _norm(attn)
    T = p.shape[-1]
    if T == 0:
        raise ValueError("Attention must contain at least one token")
    def mean_or_zero(values):
        return values.mean(-1) if values.shape[-1] else p.new_zeros(p.shape[:2])
    idx = torch.arange(T, device=p.device)
    band = ((idx[:, None] - idx[None, :]).abs() <= window).float()
    return {
        "first_token": mean_or_zero(p[..., 1:, 0]),
        "previous_token": mean_or_zero(torch.diagonal(p, -1, -2, -1)),
        "next_token": mean_or_zero(torch.diagonal(p, 1, -2, -1)),
        "self": torch.diagonal(p, 0, -2, -1).mean(-1),
        "local_window": (p * band).sum((-1, -2)) / T,
    }


def classify(scores, entropy, cfg):
    """Returns nested list [L][H] of type names."""
    L, H = entropy.shape
    out = [[None] * H for _ in range(L)]
    for l in range(L):
        for h in range(H):
            for t in TYPES[:5]:
                if scores[t][l, h] >= cfg[t]:
                    out[l][h] = t
                    break
            else:
                out[l][h] = "broad" if entropy[l, h] >= cfg["broad_entropy"] else "other"
    return out