import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.C1 = nn.Conv2d(1, 6, 5, stride=1)
        self.S2 = nn.MaxPool2d(2)
        self.C3 = nn.Conv2d(6, 16, 5, stride=1)
        self.S4 = nn.MaxPool2d(2)
        self.F5 = nn.Linear(16*5*5, 120)
        self.F6 = nn.Linear(120, 84)
        self.F7 = nn.Linear(84, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = torch.flatten(x, 1)
        if False:
            x = self.F5(x)
            x = self.F6(x)
            x = self.F7(x)
        else:
            x = self.F5(self.activation(x))
            x = self.F6(self.activation(x))
            x = self.F7(x)
        return x


if __name__ == "__main__":

    model = Net()
    print(model)

    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    target = torch.randn(1, 10)
    target = target.view(1, -1)
    print(y.shape)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training
    for i in range(5):
        y = model(x)
        loss = criterion(y, target)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

    # learning_rate = 0.01
    # for f in model.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)
