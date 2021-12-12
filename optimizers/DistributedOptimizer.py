import sys
import torch
import torch.distributed as dist


class DistributedOptimizer(torch.optim.Optimizer):

    def __init__(self, params, local_opt, defaults):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_opt = local_opt
        self.data_transferred = 0

        super(DistributedOptimizer, self).__init__(params, defaults)

    def step(self, closure):
        pass

    def avg(self, input):

        input_size = sys.getsizeof(input.storage())

        dist.reduce(input, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            input /= self.world_size
        else:
            self.data_transferred += input_size

        dist.broadcast(input, src=0)
        if self.rank == 0:
            self.data_transferred += input_size * self.world_size
