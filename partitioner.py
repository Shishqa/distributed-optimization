import torch
import torch.distributed as dist

from random import Random


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, batch_size):
    size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]

    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    loader = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)

    return loader, bsz
