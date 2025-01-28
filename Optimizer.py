import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def get_lr_lambda(d_model, n_warmup_steps, lr_factor=1.0):
    def lr_lambda(n_steps):
        return lr_factor * ((d_model ** -0.5) * min(n_steps ** -0.5, n_steps * (n_warmup_steps ** -1.5)))
    return lr_lambda

class CustomOptim:
    def __init__(self, model, lr=1e-4, beta1=0.9, beta2=0.98, eps = 1e-9, d_model=512, n_warmup_steps=4000, lr_factor=1):
        self.optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps),
        self.eps = eps
        self.lr_factor = lr_factor
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=get_lr_lambda(self.d_model, self.n_warmup_steps, self.lr_factor))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # increase step counter
        self.n_steps += 1

        # adpat learning rate
        self.scheduler.step()

        # perform optimizer step
        self.optimizer.step()

