"""Weighted spherical discrimination for the flag update."""

import math

import torch


@torch.no_grad()
def loss_and_grad(U, a_col, near, far, *, eps, lambda_neg=1.0):
    """Minimize -S_positive + lambda_neg*S_negative with frozen embeddings.

    Pair weights average within each query, then over queries with negatives.
    Weighted coordinates implement A**(1/2), avoiding a full D-by-D metric.
    A local autograd leaf includes the derivative of cosine normalization even
    when this function is called from the no_grad flag-update context.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("The squared-norm regularizer eps must be positive and finite.")
    if not math.isfinite(lambda_neg) or lambda_neg < 0:
        raise ValueError("flag_lambda_neg must be nonnegative and finite.")
    with torch.enable_grad(), torch.autocast(device_type=U.device.type, enabled=False):
        basis = U.detach().float().requires_grad_(True)
        scale = a_col.detach().float().sqrt()

        def cosine_matrix(pairs):
            query, target = (t.detach().float() for t in pairs[:2])
            eq, et = (query @ basis) * scale, (target @ basis) * scale
            eq = eq / (eq.square().sum(-1, keepdim=True) + eps).sqrt()
            et = et / (et.square().sum(-1, keepdim=True) + eps).sqrt()
            return eq @ et.T

        near_cosine = cosine_matrix(near)
        # Training uses the same query/target pool for both relations.
        far_cosine = near_cosine if near[0] is far[0] and near[1] is far[1] else cosine_matrix(far)
        positive = (near_cosine * near[2].detach().float()).sum()
        negative = (far_cosine * far[2].detach().float()).sum()
        loss = -positive + lambda_neg * negative
        gradient, = torch.autograd.grad(loss, basis)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite spherical flag loss or gradient.")
        return loss.detach(), gradient.detach(), {
            "flag_pos_cosine": positive.detach(),
            "flag_neg_cosine": negative.detach(),
            "flag_gap": (positive - negative).detach(),
        }
