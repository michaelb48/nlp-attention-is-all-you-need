import torch
from torch.optim import Adam

class CustomOptim:
    def __init__(self, model, lr=1e-4, beta1=0.9, beta2=0.98, eps = 1e-9, d_model=512, n_warmup_steps=4000, lr_factor=1):
        self.optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps),
        self.eps = eps
        self.lr_factor = lr_factor
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self):
        # calculate according to formula in paper
        lr = lr_factor * ((d_model ** -0.5) * min(n_steps ** -0.5, n_steps * (n_warmup_steps ** -1.5)))

        # update the lr for every param in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self):
        # increase step counter
        self.n_steps += 1

        # adpat learning rate
        self.lr_step()

        # perform optimizer step
        self.optimizer.step()