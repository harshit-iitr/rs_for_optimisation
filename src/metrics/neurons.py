import torch

def compute_dead_fraction(h_post):
    """
    Fraction of units with zero output on all probe inputs.
    h_post: (B, d) POST-activations (e.g. after ReLU).
    """
    h_abs_sum = torch.sum(torch.abs(h_post), dim=0)
    dead_units = (h_abs_sum == 0).sum().item()
    return dead_units / h_post.shape[-1]

def compute_dormant_fraction(h_post, threshold=0.025):
    """
    Fraction with normalized activation score < 0.025 (Sokar et al.)
    Score for neuron i: E_x[|h_i(x)|] / (1/d \sum_j E_x[|h_j(x)|])
    h_post: (B, d) POST-activations.
    """
    mean_abs_activation = torch.mean(torch.abs(h_post), dim=0) # (d,)
    layer_mean = torch.mean(mean_abs_activation) + 1e-12
    normalized_score = mean_abs_activation / layer_mean
    dormant_units = (normalized_score < threshold).sum().item()
    return dormant_units / h_post.shape[-1]
