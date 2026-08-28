import torch

class SI:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
        self.W = {}
        self.p_old = {}
        self.omega = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.W[n] = torch.zeros_like(p.data)
                self.p_old[n] = p.data.clone().detach()
                self.omega[n] = torch.zeros_like(p.data)

    def update_W(self, old_params):
        # Called AFTER optimizer.step()
        # old_params is a dict of previous step's parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta_p = p.data - old_params[n]
                # W accumulates (-grad * delta_p)
                self.W[n].add_(-p.grad.data * delta_p)

    def update_omega(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                delta_p_total = p.data - self.p_old[n]
                self.omega[n].add_(self.W[n] / (delta_p_total.pow(2) + self.epsilon))
                self.W[n].zero_()
                self.p_old[n] = p.data.clone().detach()

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.omega:
                loss += (self.omega[n] * (p - self.p_old[n]).pow(2)).sum()
        return loss
