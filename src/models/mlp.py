import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain MLP with optional LayerNorm and an optional hard spherical
    constraint on pre-activations.

    `projection` selects the constraint arm and is threaded through from the
    trainer: None/"none" for the soft-penalty and baseline arms, "tangential" for
    the true hard-constraint limit, "ste" for the Round 1-4 straight-through
    variant. The two are different algorithms and are never aliased.
    """

    def __init__(self, input_dim=784, hidden_dims=None, depth=3, width=1000,
                 output_dim=10, act_fn=nn.ReLU, use_ln=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList() if use_ln else None
        if hidden_dims is None:
            hidden_dims = [width] * depth
        in_dim = input_dim
        for h_dim in hidden_dims:
            layer = nn.Linear(in_dim, h_dim)
            nn.init.kaiming_uniform_(
                layer.weight, nonlinearity="relu" if act_fn == nn.ReLU else "leaky_relu")
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            if use_ln:
                self.lns.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        self.head = nn.Linear(in_dim, output_dim)
        nn.init.kaiming_uniform_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)
        self.act_fn = act_fn()

    def forward(self, x, return_activations=False, projection=None):
        from src.methods.rs import apply_hard_projection

        x = x.view(x.size(0), -1)
        pre_acts, post_acts = [], []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.lns is not None:
                x = self.lns[i](x)
            if projection not in (None, "none"):
                x = apply_hard_projection(x, projection)
            if return_activations:
                pre_acts.append(x)
            x = self.act_fn(x)
            if return_activations:
                post_acts.append(x)
        out = self.head(x)
        if return_activations:
            return out, pre_acts, post_acts
        return out
