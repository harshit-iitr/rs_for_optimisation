import torch

class EWC:
    def __init__(self, model, gamma=1.0):
        self.model = model
        self.gamma = gamma
        self.fisher = {}
        self.optpar = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = torch.zeros_like(p.data)
                self.optpar[n] = p.data.clone().detach()

    def update_fisher(self):
        # Assumes model.backward() has just been called on a batch
        # for empirical Fisher computation
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # Online EWC: decay old Fisher and add new empirical Fisher
                self.fisher[n] = self.gamma * self.fisher[n] + p.grad.data.pow(2)
                self.optpar[n] = p.data.clone().detach()

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.optpar[n]).pow(2)).sum()
        return loss
