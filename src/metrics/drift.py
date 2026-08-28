"""Representation drift and subspace metrics.

Extracted from the Round 1-4 training loop, where they lived inline and untested
(audit section 1.3). Two corrections carried over from the audit:

1.  ABSOLUTE drift is logged alongside relative drift. The Round 1-4 metrics
    divided by ||h_past||, which the penalty directly shrinks (radius_mean falls
    from 49.8 at lambda=0 to 34.2 at lambda=0.03), so an identical absolute
    displacement read ~45% larger in the penalised arm. The relative versions are
    kept for archive comparability but must never again be the only ones logged.

2.  Drift against a FIXED task-0 reference is logged alongside consecutive
    task-boundary drift. Round 1-4 logged only the consecutive version while the
    reports described it as preservation "across tasks".

Subspace overlap keeps k = 50 fixed across all arms so the same rank is always
compared. Both the mean principal-angle cosine (the Round 1-4 definition, kept
for comparability) and the standard projection metric (mean squared cosine) are
returned.
"""

import torch

from src.metrics.linalg import safe_svd_u

SUBSPACE_K = 50


def _flat(h):
    return h.flatten(1) if h.dim() > 2 else h


def compute_drift(h_curr, h_past, eps=1e-8):
    """Displacement of a fixed probe's representation between two checkpoints.

    h_curr, h_past: (B, d) activations of the SAME probe inputs.

    Returns absolute and relative magnitudes, split into the component along
    h_past (radial) and orthogonal to it (tangential), plus mean cosine
    similarity. Absolute quantities are the ones comparable across penalty
    strengths.
    """
    a, b = _flat(h_curr), _flat(h_past)
    delta = a - b

    b_norm_row = torch.norm(b, p=2, dim=1, keepdim=True).clamp(min=eps)
    b_hat = b / b_norm_row
    delta_rad = (delta * b_hat).sum(dim=1, keepdim=True) * b_hat
    delta_tan = delta - delta_rad

    ref = torch.norm(b) + eps
    return {
        "drift_abs": torch.norm(delta).item(),
        "drift_rad_abs": torch.norm(delta_rad).item(),
        "drift_tan_abs": torch.norm(delta_tan).item(),
        "ref_norm": torch.norm(b).item(),
        "drift_rel": (torch.norm(delta) / ref).item(),
        "drift_rad": (torch.norm(delta_rad) / ref).item(),
        "drift_tan": (torch.norm(delta_tan) / ref).item(),
        "drift_cos_sim": torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item(),
    }


def subspace_basis(h, k=SUBSPACE_K):
    """Orthonormal basis of the dominant k-dimensional activation subspace.

    Split out from the overlap so the trainer can CACHE it: the previous task's
    basis is recomputed every task otherwise, and the fixed task-0 reference
    basis would be recomputed 150 times.
    """
    return safe_svd_u(_flat(h).t(), k)


def overlap_from_bases(U_curr, U_past):
    """Principal-angle similarity between two subspaces given their bases."""
    nan = {"subspace_overlap": float("nan"), "subspace_proj_metric": float("nan"),
           "subspace_k": 0}
    if U_curr is None or U_past is None:
        return nan
    kk = min(U_curr.shape[1], U_past.shape[1])
    try:
        cos_angles = torch.linalg.svdvals(U_curr[:, :kk].t() @ U_past[:, :kk])
    except Exception:
        return nan
    return {
        # Round 1-4 definition, preserved for archive comparability.
        "subspace_overlap": cos_angles.mean().item(),
        # Standard projection metric: mean squared cosine of principal angles.
        "subspace_proj_metric": (cos_angles ** 2).mean().item(),
        "subspace_k": int(kk),
    }


def compute_subspace_overlap(h_curr, h_past, k=SUBSPACE_K):
    """Convenience wrapper. k is fixed at 50 for every arm, so the same rank is
    always compared."""
    return overlap_from_bases(subspace_basis(h_curr, k), subspace_basis(h_past, k))
