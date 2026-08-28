import torch
import math
from src.metrics.phi_rad import compute_phi_rad_tilde
from src.methods.rs import compute_rs_penalty

def test_phi_rad_tilde_random():
    d = 1000
    h = torch.randn(1000, d)
    g = torch.randn(1000, d)
    
    val = compute_phi_rad_tilde(h, g)
    assert abs(val - 1.0) < 0.15, f"Expected ~1.0, got {val}"

def test_phi_rad_tilde_parallel():
    d = 1000
    h = torch.randn(100, d)
    g = h.clone() * 2.5
    
    val = compute_phi_rad_tilde(h, g)
    assert abs(val - d) < 1e-3, f"Expected {d}, got {val}"

def test_phi_rad_tilde_orthogonal():
    d = 1000
    h = torch.randn(100, d)
    # create orthogonal g
    g = torch.randn(100, d)
    # project out h
    h_norm = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    h_hat = h / h_norm
    g_rad = torch.sum(g * h_hat, dim=-1, keepdim=True) * h_hat
    g = g - g_rad
    
    # check orthogonality
    assert torch.allclose(torch.sum(h * g, dim=-1), torch.zeros(100), atol=1e-4)
    
    val = compute_phi_rad_tilde(h, g)
    assert abs(val - 0.0) < 1e-5, f"Expected 0.0, got {val}"

def test_phi_rad_tilde_leakage():
    d = 1000
    h = torch.randn(100, d, requires_grad=True)
    
    # Dummy task loss: L_task = sum(h * c) where c is random
    c = torch.randn_like(h)
    task_loss = torch.sum(h * c)
    # Extract task gradient BEFORE penalty
    g_task = torch.autograd.grad(task_loss, h, retain_graph=True)[0]
    
    val_lambda_0 = compute_phi_rad_tilde(h.detach(), g_task.detach())
    
    # Now with penalty
    lam = 1e6
    rs_penalty = compute_rs_penalty(h)
    total_loss = task_loss + lam * rs_penalty
    
    # Normally, people might do total_loss.backward() and take h.grad
    total_loss.backward()
    g_total = h.grad
    
    # This should fail if we mistakenly pass g_total
    val_lambda_1_wrong = compute_phi_rad_tilde(h.detach(), g_total.detach())
    assert abs(val_lambda_1_wrong - val_lambda_0) > 1e-3, "Penalty gradient did not change total gradient! Test is flawed."
    
    # But if we pass g_task explicitly, it should match exactly
    val_lambda_1_correct = compute_phi_rad_tilde(h.detach(), g_task.detach())
    assert abs(val_lambda_1_correct - val_lambda_0) < 1e-7, "Penalty gradient leaked into task gradient!"
