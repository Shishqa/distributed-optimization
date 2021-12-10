import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import numpy as np

from math import ceil
from torchvision import datasets, transforms

from optimizers.DANE import DANE
from partitioner import partition_dataset


def prepare_data():
    BATCH_SIZE = 10000

    full_ds = datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    val_len = ceil(len(full_ds) * 0.15)
    split = [len(full_ds) - val_len, val_len]
    train_ds, val_ds = torch.utils.data.random_split(full_ds, split)

    train_loader, _ = partition_dataset(train_ds, BATCH_SIZE)
    val_loader, _ = partition_dataset(val_ds, BATCH_SIZE)

    test_ds = datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    test_loader, bsz = partition_dataset(test_ds, BATCH_SIZE)

    return train_loader, val_loader, test_loader, bsz


def evaluate(model, criterion, data_loader):

    avg_accuracy = []
    avg_loss = []

    model.eval()
    for data, target in data_loader:
        output = model(data)

        _, pred = torch.max(output, dim=1)
        avg_accuracy.append(torch.sum(pred == target).item() / len(pred))

        loss = criterion(output, target)
        avg_loss.append(loss.item())

    return np.mean(avg_loss), np.mean(avg_accuracy)


def run(rank, size):
    torch.manual_seed(1234)

    train_loader, val_loader, test_loader, bsz = prepare_data()

    model = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1),
    )

    criterion = nn.NLLLoss()

    optimizer = DANE(
            model.parameters(),
            local_opt=optim.SGD,
            lr=1,
            mu=0
            )

    num_batches = ceil(len(train_loader.dataset) / float(bsz))

    for epoch in range(10):

        model.train()

        def closure():
            loss = None
            for data, target in train_loader:
                output = model(data)
                if loss is None:
                    loss = criterion(output, target)
                else:
                    loss += criterion(output, target)
            loss /= num_batches
            return loss

        optimizer.step(closure)

        eval_loss, eval_acc = evaluate(model, criterion, val_loader)

        print('[{}]: Epoch {} :: Loss = {}, Accuracy = {}'.format(
            rank, epoch, eval_loss, eval_acc
        ))

    test_loss, test_acc = evaluate(model, criterion, test_loader)

    print('[{}] Finally :: Loss = {}, Accuracy = {}'.format(
        rank, test_loss, test_acc
    ))


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
