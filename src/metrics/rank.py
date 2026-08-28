"""Effective and stable rank of an activation matrix.

Definitions preserved bit-for-bit from Round 1-4 (audit section 7.3). The only
change is that both are now computed from a SINGLE singular-value decomposition
instead of two, and on the robust CPU path (see src/metrics/linalg.py).
"""

import torch

from src.metrics.linalg import safe_svdvals


def compute_ranks(A):
    """Both ranks from one decomposition.

    eff_rank    = exp(entropy of the normalized singular-value spectrum)
    stable_rank = ||A||_F^2 / ||A||_2^2
    """
    S = safe_svdvals(A)
    if S is None:
        return float("nan"), float("nan")
    p = S / (torch.sum(S) + 1e-12)
    eff = torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item()
    stable = (torch.sum(A.detach().to(torch.float64) ** 2)
              / (S[0] ** 2 + 1e-12)).item()
    return eff, stable


def compute_effective_rank(A):
    return compute_ranks(A)[0]


def compute_stable_rank(A):
    return compute_ranks(A)[1]
