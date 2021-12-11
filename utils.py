import torch
import torch.distributed as dist
import numpy as np


def evaluate(model, loss_func, data_loader):

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    with torch.no_grad():
        val_accuracy = []
        val_loss = []

        model.eval()
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            _, pred = torch.max(output, dim=1)
            val_accuracy.append(torch.sum(pred == target).item() / len(pred))

            loss = loss_func(output, target)
            val_loss.append(loss.item())

        return np.mean(val_loss), np.mean(val_accuracy)


def consensus_train(model, loss_func, opt, train_loader,
                    n_epochs=10, verbose=True):

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    train_history = {
        'acc': [],
        'loss': []
    }

    for epoch in range(n_epochs):

        model.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            def local_objective():
                output = model(data)
                return loss_func(output, target)

            # Consensus optimization
            opt.step(local_objective)

        eval_loss, eval_acc = evaluate(model, loss_func, train_loader)

        eval_loss = torch.tensor(eval_loss)
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        eval_loss /= dist.get_world_size()

        eval_acc = torch.tensor(eval_acc)
        dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)
        eval_acc /= dist.get_world_size()

        if dist.get_rank() == 0:

            if verbose:
                print('Epoch {} ::\tLoss = {},\tAccuracy = {}'.format(
                    epoch+1, eval_loss.item(), eval_acc.item()
                ))

            train_history['acc'].append(eval_acc.item())
            train_history['loss'].append(eval_loss.item())

    return train_history
