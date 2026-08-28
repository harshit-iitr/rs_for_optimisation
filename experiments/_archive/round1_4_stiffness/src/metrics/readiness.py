import torch

def compute_readiness(g_microbatches):
    """
    Computes optimization-readiness proxy.
    g_microbatches: list of 1D tensors (flattened gradients) from m=8 microbatches.
    strength = ||mean(g_i)||_2
    reliability = ||mean(g_i)||_2^2 / mean(||g_i||_2^2)
    readiness = strength * reliability
    """
    if not g_microbatches:
        return 0.0
    
    g_stack = torch.stack(g_microbatches, dim=0) # (m, D)
    mean_g = torch.mean(g_stack, dim=0) # (D,)
    
    strength = torch.norm(mean_g, p=2).item()
    
    mean_g_norm_sq = torch.sum(mean_g**2)
    g_i_norm_sq_mean = torch.mean(torch.sum(g_stack**2, dim=-1))
    
    reliability = (mean_g_norm_sq / (g_i_norm_sq_mean + 1e-12)).item()
    
    readiness = strength * reliability
    return readiness
