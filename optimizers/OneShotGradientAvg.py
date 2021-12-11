import torch
import torch.distributed as dist


class OneShotGradientAvg(torch.optim.Optimizer):

    def __init__(self, params, local_opt):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_opt = local_opt

        defaults = dict()
        super(OneShotGradientAvg, self).__init__(params, defaults)

    def step(self, closure):
        loss = closure()
        self.zero_grad()
        loss.backward()
        self.reduce_grads()
        self.local_opt.step()

    def sync_tensor(self, local):
        dist.reduce(local, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            local /= self.world_size
        dist.broadcast(local, src=0)

    def reduce_grads(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    return

                self.sync_tensor(p.grad.data)
