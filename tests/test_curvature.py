"""Tests for src/metrics/curvature.py.

Three tests:
1. Hutchinson trace on a known quadratic converges to Tr(A).
2. Power-iteration sigma_max on a known quadratic converges to max eigenvalue.
3. Smoke test on an MLP: both functions return finite positive floats.
"""

import torch
import torch.nn as nn

from src.metrics.curvature import hutchinson_trace, power_iteration_sigma_max


# ---- helpers ----

class QuadraticModel(nn.Module):
    """Model whose loss = 0.5 * w^T A w, so H = A exactly."""

    def __init__(self, A):
        super().__init__()
        d = A.shape[0]
        self.w = nn.Parameter(torch.randn(d))
        self.A = A  # not a parameter — fixed matrix

    def forward(self, x=None):
        # ignore x; the "logit" is a scalar = 0.5 * w^T A w
        return (0.5 * self.w @ self.A @ self.w).unsqueeze(0)


def _quadratic_loss(logits, targets=None):
    """Identity loss: the model output IS the loss."""
    return logits.squeeze()


# ---- tests ----

def test_hutchinson_trace_quadratic():
    """On f(w) = 0.5 * w^T A w, Tr(H) = Tr(A)."""
    torch.manual_seed(42)
    d = 20
    # Symmetric PD matrix with known trace
    M = torch.randn(d, d)
    A = M @ M.T + torch.eye(d)  # guaranteed PD
    true_trace = torch.trace(A).item()

    model = QuadraticModel(A)
    # x and y are unused by the quadratic model
    x_dummy = torch.zeros(1)
    y_dummy = torch.zeros(1)

    est = hutchinson_trace(model, _quadratic_loss, x_dummy, y_dummy,
                           n_probes=200)

    rel_err = abs(est - true_trace) / abs(true_trace)
    assert rel_err < 0.15, (
        f"Hutchinson trace {est:.4f} vs true {true_trace:.4f}, "
        f"relative error {rel_err:.3f} > 0.15")


def test_sigma_max_quadratic():
    """On f(w) = 0.5 * w^T A w, sigma_max = max eigenvalue of A."""
    torch.manual_seed(42)
    d = 20
    M = torch.randn(d, d)
    A = M @ M.T + torch.eye(d)
    true_sigma = torch.linalg.eigvalsh(A)[-1].item()

    model = QuadraticModel(A)
    x_dummy = torch.zeros(1)
    y_dummy = torch.zeros(1)

    est = power_iteration_sigma_max(model, _quadratic_loss, x_dummy, y_dummy,
                                    n_iters=50)

    rel_err = abs(est - true_sigma) / abs(true_sigma)
    assert rel_err < 0.10, (
        f"Power-iteration sigma_max {est:.4f} vs true {true_sigma:.4f}, "
        f"relative error {rel_err:.3f} > 0.10")


def test_curvature_on_mlp():
    """Smoke test: both functions return finite positive floats on an MLP."""
    torch.manual_seed(42)
    from src.models.mlp import MLP

    model = MLP(input_dim=784, width=64, depth=2, output_dim=10)
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()

    tr = hutchinson_trace(model, criterion, x, y, n_probes=10)
    sm = power_iteration_sigma_max(model, criterion, x, y, n_iters=10)

    assert isinstance(tr, float), f"hessian_trace should be float, got {type(tr)}"
    assert isinstance(sm, float), f"sigma_max should be float, got {type(sm)}"
    assert tr == tr, "hessian_trace is NaN"      # NaN check
    assert sm == sm, "sigma_max is NaN"
    assert tr > 0, f"hessian_trace should be positive, got {tr}"
    assert sm > 0, f"sigma_max should be positive, got {sm}"


def test_hutchinson_convergence_on_mlp():
    """Table 8-style convergence check: variance of the Hutchinson estimate
    decreases as the number of probes increases.

    The paper (Table 8) shows that going from 50 to 200 probes cuts relative
    error from 4.2% to 1.4%. We verify the analogous property: running the
    estimator multiple times at each probe count, the spread (max-min) should
    shrink as probe count grows.
    """
    torch.manual_seed(42)
    from src.models.mlp import MLP

    model = MLP(input_dim=784, width=64, depth=2, output_dim=10)
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()

    probe_counts = [10, 50, 200]
    n_repeats = 5
    spreads = []

    for n_probes in probe_counts:
        estimates = []
        for _ in range(n_repeats):
            est = hutchinson_trace(model, criterion, x, y, n_probes=n_probes)
            estimates.append(est)
        spread = max(estimates) - min(estimates)
        spreads.append(spread)

    # The spread should decrease (or at least not increase) as probes grow.
    # Allow a small tolerance — stochastic, so we check 200 probes < 10 probes.
    assert spreads[-1] < spreads[0], (
        f"Hutchinson estimator did not converge: spread at {probe_counts[0]} "
        f"probes = {spreads[0]:.4f}, spread at {probe_counts[-1]} probes = "
        f"{spreads[-1]:.4f}. Expected the spread to shrink."
    )
