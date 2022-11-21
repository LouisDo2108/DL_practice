import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ZFNet(nn.Module):

    def __init__(self):
        super(ZFNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, stride=2, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(self.relu(x))
        # x = self.fc2(self.relu(x))
        # x = self.fc3(x)
        return x


if __name__ == "__main__":

    model = ZFNet()
    print(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    target = torch.randn(1, 1000)
    target = target.view(1, -1)
    print(y.shape)

    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    # # Training
    # for i in range(5):
    #     y = model(x)
    #     loss = criterion(y, target)
    #     model.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())

    # learning_rate = 0.01
    # for f in model.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)
