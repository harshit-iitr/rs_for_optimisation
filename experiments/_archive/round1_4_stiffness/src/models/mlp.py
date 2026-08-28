import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=None, depth=3, width=1000, output_dim=10, act_fn=nn.ReLU, use_ln=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList() if use_ln else None
        
        if hidden_dims is None:
            hidden_dims = [width] * depth
            
        in_dim = input_dim
        for h_dim in hidden_dims:
            layer = nn.Linear(in_dim, h_dim)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu' if act_fn == nn.ReLU else 'leaky_relu')
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            if use_ln:
                self.lns.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
            
        self.head = nn.Linear(in_dim, output_dim)
        nn.init.kaiming_uniform_(self.head.weight, nonlinearity='linear')
        nn.init.zeros_(self.head.bias)
        self.act_fn = act_fn()
        
    def forward(self, x, return_activations=False, hard_projection=False):
        x = x.view(x.size(0), -1)
        
        pre_acts = []
        post_acts = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.lns is not None:
                x = self.lns[i](x)
            if hard_projection:
                from src.methods.rs import apply_hard_projection
                x = apply_hard_projection(x)
            if return_activations:
                pre_acts.append(x)
            x = self.act_fn(x)
            if return_activations:
                post_acts.append(x)
                
        out = self.head(x)
        
        if return_activations:
            return out, pre_acts, post_acts
        return out
