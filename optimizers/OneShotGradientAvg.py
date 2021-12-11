from .DistributedOptimizer import DistributedOptimizer


class OneShotGradientAvg(DistributedOptimizer):

    def __init__(self, params, local_opt):
        defaults = dict()
        super(OneShotGradientAvg, self).__init__(params, local_opt, defaults)

    def step(self, closure):
        loss = closure()
        self.zero_grad()
        loss.backward()
        self.reduce_grads()
        self.local_opt.step()

    def reduce_grads(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    return

                self.avg(p.grad.data)
