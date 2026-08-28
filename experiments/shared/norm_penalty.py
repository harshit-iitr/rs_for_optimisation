import torch
import torch.nn as nn

def norm_penalty(h: torch.Tensor) -> torch.Tensor:
    """
    Computes the activation norm penalty:
    L_norm(h) = (1/d) * (||h||_2 - sqrt(d))^2
    averaged over the batch dimension.
    
    Supports 2D (B, d) and 4D (B, C, H, W) activations by flattening.
    
    Args:
        h (torch.Tensor): Layer pre-activations of shape (batch_size, ...)
        
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Flatten everything except the batch dimension
    if h.dim() > 2:
        h_flat = h.flatten(start_dim=1)
    else:
        h_flat = h
        
    d = h_flat.shape[-1]
    # Compute L2 norm along the feature dimension
    norms = torch.norm(h_flat, p=2, dim=-1)
    target = d ** 0.5
    loss = (1.0 / d) * torch.mean((norms - target) ** 2)
    return loss

