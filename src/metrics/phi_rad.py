import torch

def compute_phi_rad_tilde(h, g_task):
    r"""
    Computes Normalized Fractional Radial Energy (\tilde{\Phi}_{rad}).
    h: activations (batch_size, d)
    g_task: gradient of TASK LOSS ONLY with respect to h (batch_size, d).
            Must not include the penalty gradient.
    
    Returns: scalar phi_rad_tilde
    """
    d = h.shape[-1]
    
    # h_hat = h / ||h||_2
    h_norm = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    h_hat = h / h_norm
    
    # g_rad = (g \cdot h_hat) h_hat
    dot_product = torch.sum(g_task * h_hat, dim=-1, keepdim=True)
    g_rad = dot_product * h_hat
    
    # ||g_rad||^2 / ||g||^2
    g_rad_norm_sq = torch.sum(g_rad**2, dim=-1)
    g_norm_sq = torch.sum(g_task**2, dim=-1).clamp(min=1e-12)
    
    phi_rad = g_rad_norm_sq / g_norm_sq
    
    # phi_rad_tilde = d * phi_rad
    phi_rad_tilde = d * phi_rad
    
    return phi_rad_tilde.mean().item()
