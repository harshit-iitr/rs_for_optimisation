import torch
import torch.nn as nn

class ReDo:
    def __init__(self, model, threshold=0.025, recycle_interval=1000):
        self.model = model
        self.threshold = threshold
        self.recycle_interval = recycle_interval
        self.step_count = 0
        
    def check_and_recycle(self, post_acts):
        self.step_count += 1
        if self.step_count % self.recycle_interval != 0:
            return
            
        with torch.no_grad():
            for i, (layer, h_post) in enumerate(zip(self.model.layers, post_acts)):
                mean_abs = h_post.abs().mean(dim=0)
                layer_mean = mean_abs.mean() + 1e-12
                score = mean_abs / layer_mean
                
                dormant_mask = score < self.threshold
                if dormant_mask.any():
                    # Reinitialize incoming
                    nn.init.kaiming_uniform_(layer.weight[dormant_mask], nonlinearity='relu')
                    nn.init.zeros_(layer.bias[dormant_mask])
                    
                    # Reinitialize outgoing
                    if i + 1 < len(self.model.layers):
                        next_layer = self.model.layers[i+1]
                        nn.init.zeros_(next_layer.weight[:, dormant_mask])
                    else:
                        next_layer = self.model.head
                        nn.init.zeros_(next_layer.weight[:, dormant_mask])
