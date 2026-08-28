import torch

class MAS:
    def __init__(self, model):
        self.model = model
        self.omega = {}
        self.optpar = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.omega[n] = torch.zeros_like(p.data)
                self.optpar[n] = p.data.clone().detach()

    def update_omega(self):
        # Assumes model output L2 norm backward has just been called
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.omega[n].add_(p.grad.data.abs())
                self.optpar[n] = p.data.clone().detach()

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.omega:
                loss += (self.omega[n] * (p - self.optpar[n]).pow(2)).sum()
        return loss
