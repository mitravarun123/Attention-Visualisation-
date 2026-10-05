import torch


def attention_rollout(attn, cumulative=False):
    """attn: [L, H, T, T]. Returns [T, T], or [L, T, T] if cumulative."""
    A = attn.float().abs().mean(1)           # head-average, [L, T, T]
    T = A.shape[-1]
    I = torch.eye(T, device=A.device, dtype=A.dtype)
    R, outs = I.clone(), []
    for l in range(A.shape[0]):
        a = 0.5 * A[l] + 0.5 * I             # residual connection
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-9)
        R = a @ R
        outs.append(R)
    return torch.stack(outs) if cumulative else R