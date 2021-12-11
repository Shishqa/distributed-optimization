import torch
from .DistributedOptimizer import DistributedOptimizer


class ADMM(DistributedOptimizer):

    def __init__(self, params, local_opt, local_n_epochs=10, lr=1, rho=0.1):
        self.local_n_epochs = local_n_epochs

        defaults = {
            'lr': lr,
            'rho': rho,
        }

        super(ADMM, self).__init__(params, local_opt, defaults)

    def step(self, closure):
        loss = closure()
        self.zero_grad()
        loss.backward()

        self.find_avg_parameter()
        self.local_optimize(closure)
        self.update_u()

    def find_avg_parameter(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if 'u' not in state:
                    state['u'] = 0

                if 'xbar' not in state:
                    state['xbar'] = torch.zeros_like(p.data)

                avg = torch.clone(p.detach())
                self.avg(avg)
                state['xbar'] = avg

    def compute_function(self, closure):
        """
        Compute function for local optimization
        """

        loss = closure()

        for group in self.param_groups:

            rho = group['rho']

            for x in group['params']:
                state = self.state[x]

                u = state['u']
                xbar = state['xbar']

                # Ensure gradients are being computed
                with torch.enable_grad():
                    # Yay, we've computed the function
                    loss += (rho / 2) * torch.linalg.norm(x - xbar + u)

        return loss

    def update_u(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'u' not in state:
                    state['u'] = torch.zeros_like(p)

                state = self.state[p]
                state['u'] = state['u'] + p.data - state['xbar']

    def local_optimize(self, closure):
        """
        Runs local optimization iterations
        """
        for _ in range(self.local_n_epochs):

            loss = self.compute_function(closure)

            self.local_opt.zero_grad()
            loss.backward()
            self.local_opt.step()
