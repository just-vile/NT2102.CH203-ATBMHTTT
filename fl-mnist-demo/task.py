# fl_mnist_demo/task.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def load_data(partition_id: int, num_partitions: int):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=transform)
    testset  = MNIST('./data', train=False, transform=transform)
    # Simple IID partition: each client gets an equal share
    n = len(trainset) // num_partitions
    indices = range(partition_id * n, (partition_id + 1) * n)
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(trainset, indices), batch_size=64, shuffle=True)
    test_loader  = DataLoader(testset, batch_size=128)
    return train_loader, test_loader

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, parameters):
    import numpy as np
    keys = list(model.state_dict().keys())
    params_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(params_dict, strict=True)

def train(model, loader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

def test(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss / len(loader), correct / n