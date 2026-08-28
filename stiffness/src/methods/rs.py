import torch
import math

class HardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):
        d = h.shape[-1]
        norm = torch.norm(h, p=2, dim=-1, keepdim=True)
        # Avoid division by zero
        norm = torch.clamp(norm, min=1e-8)
        projected = h * (math.sqrt(d) / norm)
        return projected

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output

def apply_hard_projection(h):
    """
    Applies hard hyperspherical projection with straight-through estimator.
    """
    return HardProjection.apply(h)

def compute_rs_penalty(h):
    """
    Computes the soft Radial Suppression penalty.
    h: (batch_size, d)
    Returns: scalar penalty
    L_RS(h) = (1/d) * (||h||_2 - \sqrt{d})^2
    """
    d = h.shape[-1]
    norm = torch.norm(h, p=2, dim=-1)
    penalty = (1.0 / d) * (norm - math.sqrt(d))**2
    return penalty.mean()
