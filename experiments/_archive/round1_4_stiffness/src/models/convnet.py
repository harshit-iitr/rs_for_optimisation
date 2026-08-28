import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=100, act_fn=nn.ReLU, use_ln=False):
        super().__init__()
        # 4-layer ConvNet, no BatchNorm
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.act_fn = act_fn()
        
        self.use_ln = use_ln
        if use_ln:
            self.ln1 = nn.LayerNorm([32, 32, 32]) # CIFAR is 32x32, conv1 output is 32x32
            self.ln2 = nn.LayerNorm([64, 16, 16])
            self.ln3 = nn.LayerNorm([128, 8, 8])
            self.ln4 = nn.LayerNorm([256, 4, 4])
            
        self.head = nn.Linear(256 * 2 * 2, output_dim)
        
        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        
        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu' if act_fn == nn.ReLU else 'leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_activations=False, hard_projection=False):
        pre_acts = []
        post_acts = []
        
        from src.methods.rs import apply_hard_projection
        
        layers = [
            (self.conv1, self.ln1 if self.use_ln else None),
            (self.conv2, self.ln2 if self.use_ln else None),
            (self.conv3, self.ln3 if self.use_ln else None),
            (self.conv4, self.ln4 if self.use_ln else None),
        ]
        
        for conv, ln in layers:
            x = conv(x)
            if ln is not None:
                x = ln(x)
            
            # Flatten for RS penalty matching
            flat_x = x.view(x.size(0), -1)
            
            if hard_projection:
                flat_x = apply_hard_projection(flat_x)
                
            x = flat_x.view_as(x)
                
            if return_activations:
                pre_acts.append(flat_x)
                
            x = self.act_fn(x)
            
            if return_activations:
                post_acts.append(x.view(x.size(0), -1))
                
            x = self.pool(x)
            
        x = x.view(x.size(0), -1)
        out = self.head(x)
        
        if return_activations:
            return out, pre_acts, post_acts
        return out
