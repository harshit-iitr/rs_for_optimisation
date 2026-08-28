"""Realized gradient-magnitude bookkeeping.

The isotropic control is only meaningful if we know what magnitude each arm
actually realizes per step, after clipping and after any scaling. Round 1-4
derived its "shrinkage factor" from end-of-task weight norms and applied it
BEFORE clip_grad_norm_, which renormalized it away; the achieved weight-norm
match was 0.7% against an intended 18.5% (audit section 3.3.2). Nothing in the
pipeline noticed.

Here every arm logs its realized per-step gradient norms -- global and per
parameter group -- so the match can be verified rather than assumed.
"""

import torch


def param_groups(model):
    """Ordered (name, [params]) groups. One group per layer (weight + bias),
    plus the head. The isotropic control matches per group, so the head and the
    biases are included -- Round 1-4 scaled only hidden-layer weights."""
    groups = []
    for i, layer in enumerate(model.layers):
        ps = [p for p in (layer.weight, getattr(layer, "bias", None)) if p is not None]
        groups.append((f"layer{i}", ps))
    head = model.head
    groups.append(("head", [p for p in (head.weight, getattr(head, "bias", None)) if p is not None]))
    return groups


@torch.no_grad()
def group_grad_norms(groups):
    """L2 norm of the current .grad within each group, plus the global norm."""
    norms, total_sq = [], 0.0
    for _, ps in groups:
        sq = 0.0
        for p in ps:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        norms.append(sq ** 0.5)
        total_sq += sq
    return norms, total_sq ** 0.5


@torch.no_grad()
def scale_groups_to(groups, targets, eps=1e-12):
    """Rescale each group's gradient so its norm equals the given target.

    Direction is untouched -- this is a pure magnitude intervention, which is
    exactly what an isotropic control must be.

    The returned norms are RE-MEASURED after scaling, never assumed. The
    acceptance test reads these; if they were the intended targets copied through,
    the test would be checking its own input and would pass unconditionally.
    """
    for (_, ps), target in zip(groups, targets):
        sq = 0.0
        for p in ps:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        cur = sq ** 0.5
        if cur > eps and target is not None:
            f = target / cur
            for p in ps:
                if p.grad is not None:
                    p.grad.detach().mul_(f)
    return group_grad_norms(groups)[0]


@torch.no_grad()
def scale_global_to(groups, target, eps=1e-12):
    """Rescale ALL gradients by one scalar so the global norm equals `target`."""
    _, cur = group_grad_norms(groups)
    if cur > eps and target is not None:
        f = target / cur
        for _, ps in groups:
            for p in ps:
                if p.grad is not None:
                    p.grad.detach().mul_(f)
        return target
    return cur
