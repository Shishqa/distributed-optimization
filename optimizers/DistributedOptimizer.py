import torch
import torch.distributed as dist


class DistributedOptimizer(torch.optim.Optimizer):

    def __init__(self, params, local_opt, defaults):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_opt = local_opt

        super(DistributedOptimizer, self).__init__(params, defaults)

    def step(self, closure):
        pass

    def avg(self, input):
        dist.reduce(input, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            input /= self.world_size
        dist.broadcast(input, src=0)
