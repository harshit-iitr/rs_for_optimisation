import torch

def compute_effective_rank(A):
    """
    Computes effective rank of activation matrix A (B x d).
    eff_rank = exp(-sum(p_i * log(p_i))) where p_i = \sigma_i / sum(\sigma_j)
    """
    try:
        # A: (B, d)
        # Use full_matrices=False for efficiency
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        p = S / (torch.sum(S) + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return torch.exp(entropy).item()
    except Exception:
        return float('nan')

def compute_stable_rank(A):
    """
    Computes stable rank of activation matrix A (B x d).
    stable_rank = ||A||_F^2 / ||A||_2^2
    """
    try:
        frob_norm_sq = torch.sum(A**2)
        # 2-norm is the largest singular value
        _, S, _ = torch.linalg.svd(A, full_matrices=False)
        max_sv_sq = S[0]**2
        return (frob_norm_sq / (max_sv_sq + 1e-12)).item()
    except Exception:
        return float('nan')
