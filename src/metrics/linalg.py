"""Robust SVD helpers.

The GPU (cusolver) SVD is data-dependently unstable on activation matrices. In
the unclipped arm it both raised `_LinAlgError: failed to converge` -- killing a
run seven tasks in -- and, when it fell back internally to its "more accurate"
path, slowed a task from 2.7 s to 1065 s.

Everything here therefore runs on CPU in float64: deterministic, run-to-run
reproducible, and robust on ill-conditioned input. It costs roughly 2x the GPU
path, which is a fraction of a run.

A failure is never silent. If even the CPU path fails, NaN is returned and a
counter is incremented; the trainer writes the counts into config.json, so a run
whose metrics degraded says so.
"""

import torch

FALLBACKS = {"svdvals_failed": 0, "svd_u_failed": 0}


def reset_counters():
    for k in FALLBACKS:
        FALLBACKS[k] = 0


def _cpu64(A):
    return A.detach().to("cpu", torch.float64)


def safe_svdvals(A):
    """Singular values, or None if the decomposition cannot be computed."""
    try:
        return torch.linalg.svdvals(_cpu64(A))
    except Exception:
        FALLBACKS["svdvals_failed"] += 1
        return None


def safe_svd_u(A, k=None):
    """Leading left singular vectors, or None. `A` is (rows, cols); the returned
    basis spans the dominant subspace of the column space."""
    try:
        U = torch.linalg.svd(_cpu64(A), full_matrices=False).U
    except Exception:
        FALLBACKS["svd_u_failed"] += 1
        return None
    return U[:, :k] if k is not None else U
