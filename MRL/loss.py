import math

import torch
from torch.nn import functional as F


def resolve_dims(full_dim, dims=None):
    if isinstance(full_dim, bool) or not isinstance(full_dim, int) or full_dim < 1:
        raise ValueError("full_dim must be a positive integer.")
    if dims is None:
        dims = [d for d in (16, 32, 64, 128, 256, 512) if d < full_dim] + [full_dim]
    else:
        dims = list(dims)
    if not dims or any(isinstance(d, bool) or not isinstance(d, int) or not 0 < d <= full_dim
                       for d in dims):
        raise ValueError("Each dimension must be an integer in [1, full_dim].")
    return sorted(set(dims + [full_dim]))


def info_nce(q, k, temperature=0.07):
    """One-way in-batch InfoNCE with diagonal positive pairs."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    logits = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).T / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels), logits


def Matry_infonce(a, b, temperature=0.07, nested_dims=None, *, level_weights=None):
    """Sum prefix InfoNCE as in the notebook; explicit weights are optional overrides."""
    if a.ndim != 2 or a.shape != b.shape or a.size(0) < 1:
        raise ValueError("a and b must have the same nonempty [batch, hidden_size] shape.")
    dims = resolve_dims(a.size(1), nested_dims)
    weights = [1.0] * len(dims) if level_weights is None else list(level_weights)
    if len(weights) != len(dims) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Provide one positive finite weight per resolved dimension.")
    total_loss = 0.0
    all_logits = {}
    for dim, weight in zip(dims, weights):
        loss, logits = info_nce(a[:, :dim], b[:, :dim], temperature)
        total_loss = total_loss + weight * loss
        all_logits[f"dim_{dim}"] = logits
    return total_loss, all_logits
