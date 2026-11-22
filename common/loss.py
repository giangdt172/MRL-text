"""One-way cosine InfoNCE with diagonal positives and in-batch negatives."""

import math

import torch
from torch.nn import functional as F


def info_nce(q, p, temperature=0.07):
    if q.ndim != 2 or q.shape != p.shape or q.size(0) < 1 or q.size(1) < 1:
        raise ValueError("q and p must have the same nonempty [batch, hidden_size] shape.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    # FP32 prevents normalize's epsilon underflowing under FP16 AMP.
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = F.normalize(q.float(), dim=-1) @ F.normalize(p.float(), dim=-1).T / temperature
        targets = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, targets), logits
