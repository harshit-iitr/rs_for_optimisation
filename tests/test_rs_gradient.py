import torch
import math
from src.methods.rs import compute_rs_penalty

def test_rs_penalty_gradient():
    d = 100
    B = 32
    h = torch.randn(B, d, requires_grad=True)
    
    penalty = compute_rs_penalty(h)
    penalty.backward()
    
    assert h.grad is not None
    
    # analytic gradient: (1/B) * (2/d) * (||h||_2 - \sqrt{d}) * (h / ||h||_2)
    h_norm = torch.norm(h, p=2, dim=-1, keepdim=True)
    h_hat = h / h_norm
    expected_grad = (1.0 / B) * (2.0 / d) * (h_norm - math.sqrt(d)) * h_hat
    
    assert torch.allclose(h.grad, expected_grad, atol=1e-5), "Gradient does not match analytic expectation."
