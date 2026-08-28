"""Drift and subspace metrics (audit section 3.3.3)."""
import torch
from src.metrics.drift import compute_drift, compute_subspace_overlap


def test_zero_drift_against_itself():
    h = torch.randn(64, 32)
    d = compute_drift(h, h)
    assert d["drift_abs"] < 1e-5 and d["drift_rel"] < 1e-5
    assert d["drift_cos_sim"] > 1 - 1e-5


def test_radial_and_tangential_decompose_the_displacement():
    a, b = torch.randn(64, 32), torch.randn(64, 32)
    d = compute_drift(a, b)
    assert abs(d["drift_rad_abs"] ** 2 + d["drift_tan_abs"] ** 2
               - d["drift_abs"] ** 2) < 1e-3


def test_absolute_drift_is_scale_free_where_relative_is_not():
    """The reason absolute drift must be logged. Halving the reference magnitude
    leaves the true displacement unchanged but doubles the relative metric -- and
    the penalty shrinks exactly that reference (radius 49.8 -> 34.2)."""
    b = torch.randn(64, 32)
    delta = 0.01 * torch.randn(64, 32)
    full = compute_drift(b + delta, b)
    half = compute_drift(0.5 * b + delta, 0.5 * b)
    assert abs(full["drift_abs"] - half["drift_abs"]) < 1e-5
    assert half["drift_rel"] > 1.9 * full["drift_rel"]


def test_subspace_overlap_is_one_against_itself():
    h = torch.randn(200, 64)
    o = compute_subspace_overlap(h, h)
    assert o["subspace_overlap"] > 1 - 1e-4
    assert o["subspace_proj_metric"] > 1 - 1e-4


def test_subspace_rank_is_fixed_across_widths():
    """Same k for every arm, so the same rank is always compared."""
    for width in (256, 512, 1000):
        o = compute_subspace_overlap(torch.randn(200, width), torch.randn(200, width))
        assert o["subspace_k"] == 50
