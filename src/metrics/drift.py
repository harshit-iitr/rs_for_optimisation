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


def compute_subspace_overlap(h_curr, h_past, k=SUBSPACE_K):
    """Similarity of the top-k activation subspaces.

    Columns of the (d, B) matrix span the representation; the top-k left singular
    vectors give the dominant k-dimensional subspace. The singular values of
    U_curr[:, :k]^T U_past[:, :k] are the cosines of the principal angles.

    k is fixed at 50 for every arm, so the same rank is always compared.
    """
    a, b = _flat(h_curr).t(), _flat(h_past).t()
    U_curr = torch.linalg.svd(a, full_matrices=False).U
    U_past = torch.linalg.svd(b, full_matrices=False).U
    kk = min(k, U_curr.shape[1], U_past.shape[1])
    cos_angles = torch.linalg.svdvals(U_curr[:, :kk].t() @ U_past[:, :kk])
    return {
        # Round 1-4 definition, preserved for archive comparability.
        "subspace_overlap": cos_angles.mean().item(),
        # Standard projection metric: mean squared cosine of principal angles.
        "subspace_proj_metric": (cos_angles ** 2).mean().item(),
        "subspace_k": int(kk),
    }
