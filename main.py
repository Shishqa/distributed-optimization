import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import fmnist
from optimizers.DANE import DANE
from optimizers.OneShotGradientAvg import OneShotGradientAvg
from utils import consensus_train, evaluate


def train_dane(rank, train_loader, test_loader):

    model = fmnist.FashionMNISTNet()
    criterion = nn.NLLLoss()

    opt = optim.Adam(model.parameters(), lr=3e-4)
    dane = DANE(model.parameters(), opt, lr=1, mu=1e-4)

    train_hist = consensus_train(model, criterion, dane, train_loader)

    test_loss, test_acc = evaluate(model, criterion, test_loader)
    print('[{}] Finally :: Loss = {}, Accuracy = {}'.format(
        rank, test_loss, test_acc
    ))

    return train_hist


def train_oneshot(rank, train_loader, test_loader):

    model = fmnist.FashionMNISTNet()
    criterion = nn.NLLLoss()

    opt = optim.Adam(model.parameters(), lr=3e-4)
    dane = OneShotGradientAvg(model.parameters(), opt)

    train_hist = consensus_train(model, criterion, dane, train_loader)

    test_loss, test_acc = evaluate(model, criterion, test_loader)
    print('[{}] Finally :: Loss = {}, Accuracy = {}'.format(
        rank, test_loss, test_acc
    ))

    return train_hist


def run(rank, size):

    torch.manual_seed(1234)
    train_loader, test_loader = fmnist.load()

    dane_hist = train_dane(rank, train_loader, test_loader)
    avg_hist = train_oneshot(rank, train_loader, test_loader)

    if rank == 0:
        _, ax = plt.subplots(1, 2, figsize=(16, 6))

        ax[0].plot(dane_hist['loss'], label='DANE')
        ax[0].plot(avg_hist['loss'], label='one shot averaging')
        ax[0].legend()

        ax[1].plot(dane_hist['acc'], label='DANE')
        ax[1].plot(avg_hist['acc'], label='one shot averaging')
        ax[1].legend()

        plt.show()


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
