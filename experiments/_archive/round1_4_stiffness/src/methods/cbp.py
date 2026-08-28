import torch
import torch.nn as nn
import math

class CBP:
    def __init__(self, model, replacement_rate=1e-6, maturity_threshold=100):
        self.model = model
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        
        self.utilities = []
        self.ages = []
        for layer in model.layers:
            d = layer.out_features
            self.utilities.append(torch.zeros(d, device=layer.weight.device))
            self.ages.append(torch.zeros(d, device=layer.weight.device))
            
    def update_and_replace(self, pre_acts):
        """
        Call this every step.
        pre_acts: list of pre-activations for each layer (B, d)
        """
        with torch.no_grad():
            for i, (layer, h) in enumerate(zip(self.model.layers, pre_acts)):
                B, d = h.shape
                # Track utility (e.g. moving average of activation magnitude)
                current_util = h.abs().mean(dim=0)
                self.utilities[i] = 0.99 * self.utilities[i] + 0.01 * current_util
                self.ages[i] += 1
                
                # Determine number of neurons to replace this step
                # replacement_rate is per step per neuron. 
                # Expected replacements: rate * d
                prob = self.replacement_rate * B # wait, replacement_rate usually per step.
                prob = self.replacement_rate
                
                # Sample neurons to replace
                replace_mask = (torch.rand(d, device=h.device) < prob) & (self.ages[i] > self.maturity_threshold)
                
                if replace_mask.any():
                    # Reinitialize incoming weights and bias
                    nn.init.kaiming_uniform_(layer.weight[replace_mask], nonlinearity='relu')
                    nn.init.zeros_(layer.bias[replace_mask])
                    
                    # Reinitialize outgoing weights (in the next layer, or head)
                    if i + 1 < len(self.model.layers):
                        next_layer = self.model.layers[i+1]
                        nn.init.zeros_(next_layer.weight[:, replace_mask])
                    else:
                        next_layer = self.model.head
                        nn.init.zeros_(next_layer.weight[:, replace_mask])
                        
                    # Reset utility and age
                    self.utilities[i][replace_mask] = 0.0
                    self.ages[i][replace_mask] = 0
