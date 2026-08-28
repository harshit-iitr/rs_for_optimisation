import torch
import numpy as np
import json
import os
from typing import Dict, List, Any

def compute_effective_rank(H: torch.Tensor) -> float:
    """
    Computes the effective rank of activation matrix H (N x d).
    effective_rank = exp(-sum(p_i * log(p_i))) where p_i = sigma_i / sum(sigma_j)
    """
    # H shape: (N, d)
    H_flat = H.view(H.shape[0], -1).detach().float()
    try:
        # Compute SVD (only singular values are needed)
        _, S, _ = torch.linalg.svd(H_flat, full_matrices=False)
    except Exception:
        # Fallback if SVD fails to converge
        return 1.0
    
    sum_S = torch.sum(S)
    if sum_S == 0:
        return 1.0
    p = S / sum_S
    # Filter out extremely small values to avoid nan in log
    p = p[p > 1e-10]
    entropy = -torch.sum(p * torch.log(p))
    return torch.exp(entropy).item()

def compute_dead_neuron_fraction(H: torch.Tensor) -> float:
    """
    Fraction of neurons where output is exactly 0 for >99% of samples.
    For pre-activation H, it is <= 0 for >99% of samples.
    """
    # H shape: (N, d)
    # Check if pre-activations are <= 0 (since ReLU(x) = 0 for x <= 0)
    inactive = (H <= 0.0).float()
    inactive_fraction = torch.mean(inactive, dim=0) # (d,)
    dead_neurons = (inactive_fraction > 0.99).float()
    return torch.mean(dead_neurons).item()

def compute_activation_norm_stats(H: torch.Tensor) -> Dict[str, float]:
    """
    Computes activation norm stats: mean, std, min, max of ||h||_2
    """
    # H shape: (N, d)
    norms = torch.norm(H, p=2, dim=-1) # (N,)
    return {
        "mean": torch.mean(norms).item(),
        "std": torch.std(norms).item(),
        "min": torch.min(norms).item(),
        "max": torch.max(norms).item()
    }

def compute_radial_energy(h: torch.Tensor, g: torch.Tensor) -> float:
    """
    Computes the radial energy:
    d * ||g_rad||^2 / ||g||^2 where g_rad = (g . h_hat) * h_hat
    """
    # h shape: (N, d)
    # g shape: (N, d) (gradient of loss w.r.t h)
    d = h.shape[-1]
    
    # Normalize h to unit vectors along feature dim
    h_norm = torch.norm(h, p=2, dim=-1, keepdim=True)
    h_hat = h / (h_norm + 1e-10)
    
    # Radial component of gradient: dot product of g and h_hat along feature dim
    g_dot_h_hat = torch.sum(g * h_hat, dim=-1) # (N,)
    g_rad_norm_sq = g_dot_h_hat ** 2 # (N,)
    
    # Total gradient norm squared
    g_norm_sq = torch.sum(g * g, dim=-1) # (N,)
    
    # Compute ratio per sample, avoiding division by zero
    mask = g_norm_sq > 1e-12
    radial_energy = torch.zeros_like(g_norm_sq)
    radial_energy[mask] = d * (g_rad_norm_sq[mask] / g_norm_sq[mask])
    
    return torch.mean(radial_energy).item()


class DiagnosticsTracker:
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = log_dir
        self.run_name = run_name
        self.history: List[Dict[str, Any]] = []
        os.makedirs(log_dir, exist_ok=True)
        
    def log(self, epoch: int, metrics: Dict[str, Any]):
        entry = {"epoch": epoch, **metrics}
        self.history.append(entry)
        
        # Save to JSON
        json_path = os.path.join(self.log_dir, f"{self.run_name}_diagnostics.json")
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)
            
    def save_checkpoint(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, prefix: str = ""):
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}{self.run_name}_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
