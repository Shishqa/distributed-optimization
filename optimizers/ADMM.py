import torch
import torch.distributed as dist

class ADMM(torch.optim.Optimizer):
    def __init__(self, params, local_opt, lr=1, rho=0.1):
        self.world_size = dist.get_world_size()

        defaults = {
            'local_opt': local_opt,
            'lr': lr,
            'rho': rho,
        }

        super(ADMM, self).__init__(params, defaults)

        for group in self.param_groups:
            self.reduce_params(group['params'])

    def step(self, closure):
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            rho = group['rho']
            lr = group['lr']
            opt = group['local_opt']
            params = group['params']
            self.find_avg_parameter(params)
            self.local_optimize(params, opt, closure, rho, lr)
            pass

        pass

    def find_avg_parameter(self, params):
        for p in params:
            state = self.state[p]
            if 'u' not in state:
                state['u'] = torch.zeros_like(p.data)

            if 'xbar' not in state:
                state['xbar'] = torch.zeros_like(p.data)

            avg = p.data.copy()
            self.avg(avg)
            state['xbar'] = avg

    def compute_function(self, params, closure, rho):
        """
        Compute function for local optimization
        """

        params_avg = params.copy()
        self.avg_params(params_avg)
        
        for x in params:
            state = self.state[x]

            u = state['u']
            xbar = state['xbar']

            loss = closure()

            with torch.enable_grad():  # Ensure gradients are being computed
                loss += (rho / 2) * torch.linalg.norm(x.data + xbar + u)  # Yay, we've computed the function

    def local_optimize(self, params, opt, closure, rho, lr):
        """
        Runs local optimization iterations
        """

        pass

    def avg(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size

    def avg_params(self, params):
        for p in params:
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= self.world_size