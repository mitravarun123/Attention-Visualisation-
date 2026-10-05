import torch


def head_entropy(attn, causal):
    """attn: [L, H, T, T] -> normalised entropy [L, H] in [0, 1]."""
    p = attn.float().abs()
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
    ent = -(p * torch.log(p.clamp(min=1e-12))).sum(-1)       # [L, H, T]
    T = p.shape[-1]
    if T == 0:
        raise ValueError("Attention must contain at least one token")
    if T == 1:
        return torch.zeros_like(ent[..., 0])
    n_valid = (torch.arange(1, T + 1, device=p.device) if causal
               else torch.full((T,), T, device=p.device))
    ent = ent / torch.log(n_valid.float()).clamp(min=1e-9)
    return ent[..., (1 if causal else 0):].mean(-1)