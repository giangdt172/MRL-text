"""ESE learn-to-express and learn-to-compress with an InfoNCE task loss.

Reference: https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
(AngleESETrainer). PCA targets are detached; no separate teacher is needed.
"""

import math

import torch
from torch.nn import functional as F

from common.loss import info_nce as pair_info_nce


def resolve_dims(full_dim, dims=None):
    dims = list(dims) if dims is not None else [d for d in (16, 32, 64, 128, 256, 512) if d < full_dim]
    if any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= full_dim for d in dims):
        raise ValueError("Evaluation dimensions must be integers in [1, hidden_size].")
    return sorted(set(dims + [full_dim]))


def pool_hidden(hidden, attention_mask, pooling):
    if pooling == "cls":
        return hidden[:, 0, :].float()
    if pooling != "mean":
        raise ValueError("pooling must be cls or mean.")
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)


def info_nce(first, second, temperature):
    return pair_info_nce(first, second, temperature)[0]


@torch.no_grad()
def pca_compress(embeddings, size):
    """Compress the d x d feature dependencies, as in ESE (not batch PCA)."""
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        values = embeddings.detach().float()
        dependency = (values.T @ values / math.sqrt(values.size(1))).softmax(dim=-1)
        components, singular_values, _ = torch.svd_lowrank(dependency, q=size)
        return values @ (components * singular_values.unsqueeze(0))


def compression_loss(prefix, target, temperature):
    # The reference implementation scales KL by T, rather than T squared.
    return F.mse_loss(prefix, target) + temperature * F.kl_div(
        F.log_softmax(prefix / temperature, dim=-1),
        F.softmax(target / temperature, dim=-1), reduction="batchmean")


def ese_loss(hidden_first, hidden_second, mask_first, mask_second, *,
             compression_size=128, temperature=0.07, kl_temperature=1.0, pooling="mean"):
    """Full final-layer task loss + prefix task/PCA losses at every block.

    Excludes HF hidden_states[0] (token embeddings). Intermediate block losses
    use 1/(1+ln(i)), averaged over the n-1 intermediate blocks as in AnglE.
    The final block is unweighted. Anchors and positives share one PCA target basis.
    """
    if len(hidden_first) != len(hidden_second) or len(hidden_first) < 2:
        raise ValueError("Expected embedding states followed by at least one Transformer block.")
    full_dim = hidden_first[-1].size(-1)
    if not 1 <= compression_size <= full_dim:
        raise ValueError("ese_compression_size must be in [1, hidden_size].")
    if any(not math.isfinite(t) or t <= 0 for t in (temperature, kl_temperature)):
        raise ValueError("Temperatures must be positive and finite.")
    n_layers = len(hidden_first) - 1
    # All loss and SVD arithmetic stays in FP32 even when the encoder uses AMP.
    with torch.autocast(device_type=mask_first.device.type, enabled=False):
        full_loss = hidden_first[-1].new_zeros((), dtype=torch.float32)
        express_loss = full_loss.clone()
        compress_loss = full_loss.clone()
        for layer in range(1, n_layers + 1):
            first = pool_hidden(hidden_first[layer], mask_first, pooling)
            second = pool_hidden(hidden_second[layer], mask_second, pooling)
            if layer == n_layers:
                full_loss = info_nce(first, second, temperature)
                weight = 1.0
            else:
                weight = 1.0 / ((1.0 + math.log(layer)) * (n_layers - 1))
            prefix_first, prefix_second = first[:, :compression_size], second[:, :compression_size]
            express_loss = express_loss + weight * info_nce(prefix_first, prefix_second, temperature)
            both = torch.cat((first, second), dim=0)
            target = pca_compress(both, compression_size)
            compress_loss = compress_loss + weight * compression_loss(
                torch.cat((prefix_first, prefix_second), dim=0), target, kl_temperature)
        total = full_loss + express_loss + compress_loss
    return total, {"full_task_loss": full_loss.detach(), "express_loss": express_loss.detach(),
                   "compress_loss": compress_loss.detach()}
