import torch

def compute_activation_radius(h):
    """
    Returns mean and std of ||h||_2
    h: (B, d) pre-activations
    """
    norm = torch.norm(h, p=2, dim=-1)
    return norm.mean().item(), norm.std().item()

def compute_weight_norm(weight):
    """
    weight: tensor
    """
    return torch.norm(weight, p='fro').item()

def compute_grad_norm(weight_grad):
    if weight_grad is None:
        return 0.0
    return torch.norm(weight_grad, p='fro').item()
