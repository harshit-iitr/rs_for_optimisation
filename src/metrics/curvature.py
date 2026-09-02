"""Curvature diagnostics: Hessian trace and top eigenvalue.

Hessian trace by Hutchinson's method with Rademacher probes.
Top eigenvalue (sigma_max) by power iteration.

Both use the standard double-grad trick for Hessian-vector products:
    grad(loss, params, create_graph=True) -> flat_grad
    grad(flat_grad @ v, params) -> Hv

Convergence of the Hutchinson estimator is validated in the paper's Table 8:
50 Rademacher probes achieve <5% relative error vs 500 probes.
"""

import torch


def _hvp(loss, params, vec):
    """Compute Hessian-vector product Hv via double backward.

    Parameters
    ----------
    loss : scalar tensor with grad graph attached (create_graph=True)
    params : list of parameter tensors
    vec : list of tensors, same shapes as params

    Returns
    -------
    list of tensors (same shapes as params): H @ v
    """
    grads = torch.autograd.grad(loss, params, create_graph=True)
    # dot product of grads with vec
    dot = sum((g * v).sum() for g, v in zip(grads, vec))
    hvp = torch.autograd.grad(dot, params, retain_graph=True)
    return [h.detach() for h in hvp]


def hutchinson_trace(model, loss_fn, x, y, n_probes=50, fwd_kwargs=None):
    """Estimate Tr(H) of the loss Hessian w.r.t. model parameters.

    Uses Hutchinson's stochastic trace estimator:
        Tr(H) = E_z[z^T H z]  where z ~ Rademacher(±1)

    Parameters
    ----------
    model : nn.Module
    loss_fn : callable(logits, targets) -> scalar
    x, y : input batch and targets
    n_probes : number of Rademacher probe vectors
    fwd_kwargs : dict of extra kwargs passed to model.forward()

    Returns
    -------
    float : estimated Hessian trace
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}

    params = [p for p in model.parameters() if p.requires_grad]

    # Single forward + grad with graph retained for all probes
    model.zero_grad()
    out = model(x, **fwd_kwargs)
    loss = loss_fn(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    trace_sum = 0.0
    for _ in range(n_probes):
        # Rademacher random vector: ±1 with equal probability
        z = [torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype) * 2 - 1
             for p in params]

        # z^T (H z) via double backward
        dot = sum((g * v).sum() for g, v in zip(grads, z))
        hvp = torch.autograd.grad(dot, params, retain_graph=True)

        # z^T H z = sum_i z_i * (Hv)_i
        trace_est = sum((zi * hvi.detach()).sum().item()
                        for zi, hvi in zip(z, hvp))
        trace_sum += trace_est

    model.zero_grad()
    return trace_sum / n_probes


def power_iteration_sigma_max(model, loss_fn, x, y, n_iters=20,
                              fwd_kwargs=None):
    """Estimate the top eigenvalue of the loss Hessian via power iteration.

    Computes the largest eigenvalue (spectral norm) by iterating:
        v <- H v / ||H v||
    and returning the Rayleigh quotient v^T H v.

    Parameters
    ----------
    model : nn.Module
    loss_fn : callable(logits, targets) -> scalar
    x, y : input batch and targets
    n_iters : number of power iteration steps
    fwd_kwargs : dict of extra kwargs passed to model.forward()

    Returns
    -------
    float : estimated top eigenvalue (sigma_max)
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}

    params = [p for p in model.parameters() if p.requires_grad]

    # Single forward + grad with graph retained
    model.zero_grad()
    out = model(x, **fwd_kwargs)
    loss = loss_fn(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Initialize random unit vector
    v = [torch.randn_like(p) for p in params]
    v_norm = sum((vi ** 2).sum() for vi in v) ** 0.5
    v = [vi / v_norm for vi in v]

    for _ in range(n_iters):
        # Hv via double backward
        dot = sum((g * vi).sum() for g, vi in zip(grads, v))
        hv = torch.autograd.grad(dot, params, retain_graph=True)
        hv = [h.detach() for h in hv]

        # Normalize
        hv_norm = sum((h ** 2).sum() for h in hv) ** 0.5
        if hv_norm < 1e-12:
            model.zero_grad()
            return 0.0
        v = [h / hv_norm for h in hv]

    # Rayleigh quotient: v^T H v
    dot = sum((g * vi).sum() for g, vi in zip(grads, v))
    hv = torch.autograd.grad(dot, params, retain_graph=True)
    sigma = sum((vi * hvi.detach()).sum().item() for vi, hvi in zip(v, hv))

    model.zero_grad()
    return sigma
