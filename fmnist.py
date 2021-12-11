import torch.nn as nn
from torchvision import datasets, transforms
from partitioner import partition_dataset


class FashionMNISTNet(nn.Module):

    def __init__(self, input_shape=28*28, num_classes=10):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_shape, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input):
        output = self.model(input)
        return output


def load(batch_size=10000):

    train_ds = datasets.FashionMNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    test_ds = datasets.FashionMNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    train_loader, _ = partition_dataset(train_ds, batch_size)
    test_loader, _ = partition_dataset(test_ds, batch_size)

    return train_loader, test_loader
