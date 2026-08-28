import torch
import torch.nn as nn
import math

def apply_shrink_and_perturb(model, shrink=0.8, noise_sigma=0.01):
    """
    Shrink and Perturb at task boundary.
    Shrinks weights towards zero by `shrink` and adds Gaussian noise.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.data.mul_(shrink)
            p.data.add_(torch.randn_like(p.data) * noise_sigma)

def compute_l2_init_penalty(model, initial_model):
    """
    L2 regularization towards initial parameters.
    """
    penalty = 0.0
    for p, p0 in zip(model.parameters(), initial_model.parameters()):
        penalty += torch.sum((p - p0.detach())**2)
    return penalty

def compute_er_penalty(h):
    """
    Proxy for He et al's ER penalty. Promotes feature diversity by penalizing
    the squared Frobenius norm of the off-diagonal covariance of pre-activations.
    h: (B, d)
    """
    B, d = h.shape
    h_centered = h - h.mean(dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / (B - 1 + 1e-12)
    
    # Penalize off-diagonal elements
    diag = torch.diag(torch.diag(cov))
    off_diag = cov - diag
    penalty = torch.sum(off_diag**2)
    
    # Also we want to prevent collapse of variances, so penalize small diagonal elements?
    # Actually, minimizing ||Cov||_F^2 / ||Cov||_1^2 or similar is complex.
    # We will use this off-diagonal penalty as a generic representation regularizer.
    return penalty
