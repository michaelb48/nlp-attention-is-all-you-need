import torch

class CustomOptim:
    def __init__(self, optimizer, lr=1e-4, beta1=0.9, beta2=0.98, eps = 1e-9, d_model=512, n_warmup_steps=4000, lr_factor=1):
        self.optimizer = optimizer
        self.eps = eps
        self.lr_factor = lr_factor
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return self.lr_factor * ((self.d_model ** -0.5) * min(self.n_steps ** -0.5, self.n_steps * (self.n_warmup_steps ** -1.5)))
        
    def lr_step(self):
        # calculate according to formula in paper
        lr = self.get_lr()

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

    def state_dict(self):
        self.optimizer.sate_dict()