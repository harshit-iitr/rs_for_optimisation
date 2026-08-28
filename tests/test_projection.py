"""The two hard-constraint arms are different algorithms and must stay distinct."""
import math
import torch
from src.methods.rs import apply_hard_projection


def _sphere(h, d):
    return torch.allclose(h.norm(dim=-1), torch.full((h.shape[0],), math.sqrt(d),
                                                     dtype=h.dtype), atol=1e-5)


def test_both_modes_land_on_the_sphere():
    for mode in ("tangential", "ste"):
        h = torch.randn(16, 64, requires_grad=True)
        assert _sphere(apply_hard_projection(h, mode), 64)


def test_tangential_kills_the_radial_gradient_component():
    h = torch.randn(16, 64, dtype=torch.double, requires_grad=True)
    y = apply_hard_projection(h, "tangential")
    g = torch.autograd.grad((y * torch.randn_like(y)).sum(), h)[0]
    h_hat = h.detach() / h.detach().norm(dim=-1, keepdim=True)
    assert (g * h_hat).sum(-1).abs().max() < 1e-10


def test_ste_preserves_the_radial_gradient_component():
    """The Round 1-4 arm is NOT the constraint limit; it passes the radial
    component straight through. This test exists so the two are never aliased."""
    h = torch.randn(16, 64, dtype=torch.double, requires_grad=True)
    up = torch.randn_like(h)
    g = torch.autograd.grad((apply_hard_projection(h, "ste") * up).sum(), h)[0]
    assert torch.allclose(g, up)
    h_hat = h.detach() / h.detach().norm(dim=-1, keepdim=True)
    assert (g * h_hat).sum(-1).abs().max() > 1e-6


def test_tangential_jacobian_is_exact():
    h = torch.randn(4, 16, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: apply_hard_projection(x, "tangential"), (h,), eps=1e-6, atol=1e-8)
