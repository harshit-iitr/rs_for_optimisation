"""SVD robustness (the failure that killed the first unclipped S1 attempt).

The GPU path raised `_LinAlgError: failed to converge` on an ill-conditioned
activation matrix seven tasks into a 150-task run, and its internal fallback
slowed a task from 2.7 s to 1065 s. These tests pin the replacement behaviour:
degenerate input degrades to NaN with a counter, never an exception.
"""
import math

import torch

from src.metrics.drift import overlap_from_bases, subspace_basis
from src.metrics.linalg import FALLBACKS, reset_counters, safe_svd_u, safe_svdvals
from src.metrics.rank import compute_ranks


def test_ill_conditioned_input_does_not_raise():
    """Rank-1 and near-singular matrices are exactly what showed up in the
    unclipped arm."""
    for A in (torch.ones(200, 64),
              torch.randn(200, 1) @ torch.randn(1, 64),
              torch.zeros(200, 64)):
        assert safe_svdvals(A) is not None or True     # must not raise
        compute_ranks(A)                               # must not raise
        subspace_basis(A)                              # must not raise


def test_non_finite_input_degrades_to_nan_and_counts():
    reset_counters()
    A = torch.full((100, 32), float("nan"))
    eff, stable = compute_ranks(A)
    assert math.isnan(eff) and math.isnan(stable)
    assert FALLBACKS["svdvals_failed"] >= 1, "a failure must be counted, not silent"
    reset_counters()
    assert subspace_basis(A) is None
    assert FALLBACKS["svd_u_failed"] >= 1


def test_overlap_with_a_missing_basis_is_nan_not_a_crash():
    o = overlap_from_bases(None, subspace_basis(torch.randn(200, 64)))
    assert math.isnan(o["subspace_overlap"])


def test_cached_basis_matches_recomputation():
    """The trainer caches the previous task's basis instead of recomputing it;
    the cached path must give the identical answer."""
    a, b = torch.randn(300, 128), torch.randn(300, 128)
    Ua, Ub = subspace_basis(a), subspace_basis(b)
    from src.metrics.drift import compute_subspace_overlap
    direct = compute_subspace_overlap(a, b)
    cached = overlap_from_bases(Ua, Ub)
    assert abs(direct["subspace_overlap"] - cached["subspace_overlap"]) < 1e-12


def test_ranks_are_deterministic_across_calls():
    """float64 CPU, so a paper metric is reproducible run to run."""
    A = torch.randn(500, 200)
    assert compute_ranks(A) == compute_ranks(A)
