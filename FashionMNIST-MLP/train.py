import torch
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(lr, batch_size, epochs):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    wandb.init(
        project="fashion_mnist",
        name=f"lr:{lr}, bs:{batch_size}, epochs:{epochs}",
        config={'lr': lr, "epochs": epochs, "batch_size": batch_size},
    )
    model = Model().to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_dataloader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                predictions = logits.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += len(y)
        accuracy = (correct / total) * 100

        wandb.log({"loss": avg_loss, "accuracy": accuracy, "epoch": epoch})
    wandb.finish()


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

if __name__ == "__main__":
    train_model(lr=args.lr, batch_size=args.batch_size, epochs=args.epochs)
