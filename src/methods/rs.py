"""The radial-suppression penalty and its hard-constraint limits.

L_RS(h) = (1/d) * (||h||_2 - sqrt(d))^2, averaged over the batch, applied to
pre-activations. Definition is preserved bit-for-bit from Round 1-4 so that
archived runs remain comparable (audit section 7.3).

Two distinct hard-constraint arms are provided, because they are NOT the same
experiment and Round 1-4 conflated them:

  * `tangential` -- the exact differential of the projection map. The radial
    gradient component is annihilated, which is what "enforcing the constraint"
    actually means. This is the true lambda -> infinity limit.
  * `ste` -- the straight-through estimator used in Round 1-4, which passes the
    radial gradient component through untouched. It is a different algorithm.
    Retained so the two can be distinguished rather than silently swapped.
"""

import math

import torch


class HardProjectionSTE(torch.autograd.Function):
    """Projection onto the sphere with a straight-through backward pass.

    This is the Round 1-4 behaviour. It does NOT implement the constraint's
    gradient geometry: the radial component of the gradient survives in full.
    Kept as an explicitly-named arm, not as the default.
    """

    @staticmethod
    def forward(ctx, h):
        d = h.shape[-1]
        norm = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        return h * (math.sqrt(d) / norm)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class HardProjectionTangential(torch.autograd.Function):
    """Projection onto the sphere with its exact Jacobian.

    Forward:   y = c * h / ||h||,  c = sqrt(d)
    Jacobian:  dy/dh = (c / ||h||) * (I - h_hat h_hat^T)

    so the backward pass removes the radial component of the incoming gradient
    and rescales. This is a true Riemannian retraction step: the constraint is
    enforced and the gradient respects it.
    """

    @staticmethod
    def forward(ctx, h):
        d = h.shape[-1]
        norm = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        h_hat = h / norm
        ctx.save_for_backward(h_hat, norm)
        ctx.c = math.sqrt(d)
        return h_hat * ctx.c

    @staticmethod
    def backward(ctx, grad_output):
        h_hat, norm = ctx.saved_tensors
        radial = (grad_output * h_hat).sum(dim=-1, keepdim=True) * h_hat
        return (ctx.c / norm) * (grad_output - radial)


def apply_hard_projection(h, mode="tangential"):
    if mode == "tangential":
        return HardProjectionTangential.apply(h)
    if mode == "ste":
        return HardProjectionSTE.apply(h)
    raise ValueError(f"unknown hard-projection mode: {mode!r}")


def compute_rs_penalty(h):
    """L_RS(h) = (1/d) * (||h||_2 - sqrt(d))^2, batch mean.

    h: (batch_size, d) pre-activations.
    Preserved bit-for-bit from Round 1-4. Covered by tests/test_rs_gradient.py.
    """
    d = h.shape[-1]
    norm = torch.norm(h, p=2, dim=-1)
    penalty = (1.0 / d) * (norm - math.sqrt(d)) ** 2
    return penalty.mean()
