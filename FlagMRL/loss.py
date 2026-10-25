import math

import torch
from torch import nn
from torch.nn import functional as F

from .encoder_loss import encoder_losses
from .flag_math import flag_loss_and_grad, orthogonality_error, stiefel_step
from .pairs import build_pair_weights
from .trace_ratio import trace_ratio_loss_and_grad


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


class FlagProjector(nn.Module):
    """FP32 orthonormal basis, updated explicitly rather than by AdamW.

    a_j = number of supervised prefixes containing coordinate j / number of levels.
    Flag objectives use x @ U @ diag(sqrt(a)); encoder loss uses x @ U.
    """

    def __init__(self, full_dim, dims=None, init="random", lr=0.1, eps=1e-4,
                 lambda_neg=1.0, retraction="svd", seed=42):
        super().__init__()
        self.dims = resolve_dims(full_dim, dims)
        for name, value, positive in (("lr", lr, False), ("eps", eps, True),
                                      ("lambda_neg", lambda_neg, False)):
            if not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
        if init not in {"identity", "random", "spherical"}:
            raise ValueError("init must be identity, random, or spherical.")
        if retraction not in {"svd", "newton_schulz"}:
            raise ValueError("retraction must be svd or newton_schulz.")
        self.lr, self.eps = lr, eps
        self.lambda_neg, self.retraction = lambda_neg, retraction
        if init == "identity":
            basis = torch.eye(full_dim, dtype=torch.float32)
        else:
            generator = torch.Generator().manual_seed(seed)
            basis = torch.linalg.qr(torch.randn(full_dim, full_dim, generator=generator)).Q
        coordinates = torch.arange(full_dim)
        weights = torch.stack([(coordinates < dim).float() for dim in self.dims]).mean(0)
        self.register_buffer("U", basis.contiguous())
        self.register_buffer("a_col", weights)

    def _apply(self, fn, recurse=True):
        # Keep the original FP32 values when a parent is cast to half/bfloat16.
        basis, weights = self.U, self.a_col
        super()._apply(fn, recurse=recurse)
        self.U = basis.to(device=self.U.device, dtype=torch.float32)
        self.a_col = weights.to(device=self.a_col.device, dtype=torch.float32)
        return self

    def forward(self, embeddings, dim=None, normalize=False):
        dim = self.U.size(1) if dim is None else dim
        if isinstance(dim, bool) or not isinstance(dim, int) or not 0 < dim <= self.U.size(1):
            raise ValueError("Projection dimension must be in [1, full_dim].")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            # Clone protects an outstanding encoder graph from subsequent basis updates.
            projected = embeddings.float() @ self.U[:, :dim].detach().clone()
            return F.normalize(projected, dim=-1) if normalize else projected

    @torch.no_grad()
    def step(self, q, p, target, pos_ids=None, far_mode="uniform", far_tau=0.1,
             *, objective="spherical"):
        if objective not in {"spherical", "trace_ratio"}:
            raise ValueError(f"Unknown flag objective: {objective}")
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, p = F.normalize(q.detach().float(), dim=-1), F.normalize(p.detach().float(), dim=-1)
            near, far, valid = build_pair_weights(q, p, target, pos_ids, far_mode, far_tau)
            if not valid:
                return {"flag_skipped": 1.0}
            if objective == "trace_ratio":
                loss, positive, negative, gradient = trace_ratio_loss_and_grad(
                    self.U, self.a_col, (q, p, near), (q, p, far), self.eps)
                metrics = {"flag_pos_scatter": positive, "flag_neg_scatter": negative,
                           "flag_trace_ratio": -loss}
            else:
                loss, positive, negative, gradient = flag_loss_and_grad(
                    self.U, self.a_col, (q, p, near), (q, p, far), self.eps, self.lambda_neg)
                metrics = {"flag_gap": positive - negative,
                           "flag_pos_cosine": positive, "flag_neg_cosine": negative}
            basis, norm = stiefel_step(self.U, gradient, self.lr, self.retraction)
            self.U.copy_(basis)
            return {"flag_loss": loss, **metrics,
                    "flag_gnorm": norm, "flag_orth_err": orthogonality_error(self.U),
                    "flag_skipped": 0.0}


class FlagMRLLoss(nn.Module):
    flag_objective = "spherical"

    def __init__(self, projector, temperature=0.07, update_every=1, level_weights=None,
                 lambda_cross=0.0, far_mode="uniform", far_tau=0.1):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        if isinstance(update_every, bool) or not isinstance(update_every, int) or update_every < 1:
            raise ValueError("update_every must be a positive integer.")
        if not math.isfinite(lambda_cross) or lambda_cross < 0:
            raise ValueError("lambda_cross must be nonnegative and finite.")
        if far_mode not in {"uniform", "hard"} or not math.isfinite(far_tau) or far_tau <= 0:
            raise ValueError("Invalid negative weighting mode or temperature.")
        count = len(projector.dims)
        weights = [1 / count] * count if level_weights is None else list(level_weights)
        if (len(weights) != count or any(not math.isfinite(w) or w <= 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-6)):
            raise ValueError(f"level_weights must contain {count} positive weights summing to one.")
        self.projector, self.temperature = projector, temperature
        self.update_every, self.weights = update_every, weights
        self.lambda_cross, self.far_mode, self.far_tau = lambda_cross, far_mode, far_tau
        self.register_buffer("step_idx", torch.zeros((), dtype=torch.long))

    def forward(self, q, p, target=None, pos_ids=None):
        if (q.ndim != 2 or p.ndim != 2 or q.size(0) < 1 or p.size(0) < 1
                or q.size(1) != self.projector.U.size(0) or p.size(1) != q.size(1)):
            raise ValueError("q and p must be nonempty [batch, full_dim] matrices.")
        if target is None:
            if q.size(0) != p.size(0):
                raise ValueError("Unequal query/target counts require explicit positive indices.")
            target = torch.arange(q.size(0), device=q.device)
        if (target.dtype != torch.long or target.shape != (q.size(0),)
                or target.min() < 0 or target.max() >= p.size(0)):
            raise ValueError("target must give a valid int64 positive index per query.")
        metrics = {"flag_skipped": 1.0}
        if self.training and int(self.step_idx) % self.update_every == 0:
            metrics = self.projector.step(q, p, target, pos_ids, self.far_mode, self.far_tau,
                                          objective=self.flag_objective)
        with torch.autocast(device_type=q.device.type, enabled=False):
            basis = self.projector.U.detach().clone()
            loss, encoder_metrics = encoder_losses(
                q.float() @ basis, p.float() @ basis, target, self.projector.dims,
                self.weights, self.temperature, lambda_cross=self.lambda_cross)
        if self.training:
            self.step_idx.add_(1)
        return loss, {**metrics, **encoder_metrics}
