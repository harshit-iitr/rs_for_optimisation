import torch
import torch.nn as nn
import math

class ContinualBackpropResetter:
    def __init__(self, eta: float = 0.01, threshold: float = 0.01):
        """
        Continual Backpropagation resetter based on Dohare et al. (2024).
        
        Args:
            eta (float): Decay rate for running average utility.
            threshold (float): Utility threshold below which neurons are reset.
        """
        self.eta = eta
        self.threshold = threshold
        self.utilities = {} # Map from layer name to tensor of utilities

    @torch.no_grad()
    def update_and_reset(self, model: nn.Module, optimizer: torch.optim.Optimizer, activations: dict):
        """
        Updates utility of neurons and resets those below threshold.
        
        Args:
            model (nn.Module): The model containing target layers.
            optimizer (torch.optim.Optimizer): The optimizer (to reset states).
            activations (dict): A dictionary mapping layer names to their outputs (post-activation/ReLU preferred).
        """
        # We need mapping of layers to find incoming and outgoing connections
        # Let's assume we reset neurons in layers:
        # e.g., for MLP: 'fc1' and 'fc2'
        # Incoming layer: layer itself (e.g. fc1). Outgoing layer: next linear layer (e.g. fc2).
        
        # We define target layers and their next layers manually or detect them.
        # For simplicity, let's define the sequential connections in SharedMLP/DeepMLP/SimpleCNN.
        # Let's write a generic mapper for the layers we support.
        
        reset_specs = []
        if hasattr(model, 'fc1') and hasattr(model, 'fc2') and hasattr(model, 'fc3'):
            # This is DeepMLP (fc1 -> fc2 -> fc3 -> fc4)
            if hasattr(model, 'fc4'):
                reset_specs = [
                    {"name": "h1", "layer": model.fc1, "next_layer": model.fc2},
                    {"name": "h2", "layer": model.fc2, "next_layer": model.fc3},
                    {"name": "h3", "layer": model.fc3, "next_layer": model.fc4}
                ]
            else: # SharedMLP (fc1 -> fc2 -> fc3)
                reset_specs = [
                    {"name": "h1", "layer": model.fc1, "next_layer": model.fc2},
                    {"name": "h2", "layer": model.fc2, "next_layer": model.fc3}
                ]
        elif hasattr(model, 'conv1') and hasattr(model, 'conv2') and hasattr(model, 'conv3') and hasattr(model, 'fc_hidden'):
            # This is SimpleCNN (conv1 -> conv2 -> conv3 -> fc_hidden -> head)
            reset_specs = [
                {"name": "conv1", "layer": model.conv1, "next_layer": model.conv2, "is_conv": True},
                {"name": "conv2", "layer": model.conv2, "next_layer": model.conv3, "is_conv": True},
                {"name": "conv3", "layer": model.conv3, "next_layer": model.fc_hidden, "is_conv_to_fc": True},
                {"name": "fc_hidden", "layer": model.fc_hidden, "next_layer": None, "is_multi_head": True} # next_layer depends on task
            ]
            
        for spec in reset_specs:
            name = spec["name"]
            layer = spec["layer"]
            next_layer = spec["next_layer"]
            is_conv = spec.get("is_conv", False)
            is_conv_to_fc = spec.get("is_conv_to_fc", False)
            is_multi_head = spec.get("is_multi_head", False)
            
            act = activations.get(name)
            if act is None:
                continue
                
            # Compute average absolute activation value over batch and spatial dims if conv
            if is_conv or is_conv_to_fc:
                # act shape: (B, C, H, W)
                mean_act = torch.mean(torch.abs(act), dim=(0, 2, 3)) # (C,)
            else:
                # act shape: (B, d)
                mean_act = torch.mean(torch.abs(act), dim=0) # (d,)
                
            # Update running average utility
            if name not in self.utilities:
                self.utilities[name] = mean_act.clone().detach()
            else:
                self.utilities[name] = (1.0 - self.eta) * self.utilities[name] + self.eta * mean_act
                
            # Find neurons below threshold
            dead_mask = self.utilities[name] < self.threshold
            num_dead = torch.sum(dead_mask).item()
            if num_dead == 0:
                continue
                
            # Reset dead neurons
            device = dead_mask.device
            indices_to_reset = torch.where(dead_mask)[0]
            
            for idx in indices_to_reset:
                idx = idx.item()
                # 1. Reset incoming weights of layer
                # weight shape: (out_channels, in_channels, ...)
                fan_in = layer.weight.shape[1]
                if is_conv or is_conv_to_fc:
                    # Conv weight shape: (out_channels, in_channels, k_h, k_w)
                    fan_in *= layer.weight.shape[2] * layer.weight.shape[3]
                    
                std = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(layer.weight.data[idx], -std, std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data[idx])
                    
                # Reset optimizer state for incoming weights
                self._reset_opt_slice(optimizer, layer.weight, idx, dim=0)
                if layer.bias is not None:
                    self._reset_opt_slice(optimizer, layer.bias, idx, dim=0)
                    
                # 2. Zero outgoing weights of layer
                if next_layer is not None:
                    if is_conv:
                        # next_layer is Conv: shape (out, in, kh, kw)
                        next_layer.weight.data[:, idx] = 0.0
                        self._reset_opt_slice(optimizer, next_layer.weight, idx, dim=1)
                    elif is_conv_to_fc:
                        # next_layer is Linear: shape (out, in). But spatial dim flat!
                        # The features from conv3 are flattened: C * H * W.
                        # For channel idx, the corresponding columns in linear layer are:
                        # [idx * H * W : (idx + 1) * H * W]
                        # Let's find H * W
                        spatial_dim = next_layer.weight.shape[1] // layer.weight.shape[0]
                        start_col = idx * spatial_dim
                        end_col = (idx + 1) * spatial_dim
                        next_layer.weight.data[:, start_col:end_col] = 0.0
                        for col in range(start_col, end_col):
                            self._reset_opt_slice(optimizer, next_layer.weight, col, dim=1)
                    else:
                        # next_layer is Linear: shape (out, in)
                        next_layer.weight.data[:, idx] = 0.0
                        self._reset_opt_slice(optimizer, next_layer.weight, idx, dim=1)
                elif is_multi_head:
                    # Reset all active heads' outgoing connections for this neuron
                    # For Split CIFAR-10, model.heads is a ModuleList of Linear layers
                    if hasattr(model, 'heads'):
                        for head in model.heads:
                            head.weight.data[:, idx] = 0.0
                            self._reset_opt_slice(optimizer, head.weight, idx, dim=1)

            # Reset utility of reset neurons to default starting utility (e.g. median of active ones, or threshold * 2)
            active_median = torch.median(self.utilities[name][~dead_mask]).item() if torch.sum(~dead_mask) > 0 else self.threshold * 2.0
            self.utilities[name][dead_mask] = max(active_median, self.threshold * 2.0)

    def _reset_opt_slice(self, optimizer: torch.optim.Optimizer, param: torch.Tensor, idx: int, dim: int):
        """
        Zeros the optimizer states (e.g., Adam exp_avg, exp_avg_sq) corresponding to a slice of param.
        """
        if param in optimizer.state:
            state = optimizer.state[param]
            for key in ['exp_avg', 'exp_avg_sq']:
                if key in state:
                    tensor = state[key]
                    if dim == 0:
                        tensor[idx] = 0.0
                    elif dim == 1:
                        tensor[:, idx] = 0.0
