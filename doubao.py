import torch
import torch.nn as nn

class DNet(nn.Module):
    def __init__(self):
        # super(Net, self).__init__()
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 308, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # 展平张量以供全连接层使用
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class ENet(nn.Module):
    def __init__(self):
        # super(Net, self).__init__()
        super().__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(16*32*38, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):# 修改为两类输入
        x1,x2=x
        x1 = self.pool1(self.relu1(self.conv1(x1)))
        x1 = self.pool2(self.relu1(self.conv2(x1)))
        x2 = self.pool1(self.relu2(self.conv1(x2)))
        x2 = self.pool2(self.relu2(self.conv2(x2)))
        x3=torch.concat((x1,x2),dim=-1)
        # 展平张量以供全连接层使用
        x3 = x3.view(x3.size(0), -1)
        x3 = self.relu3(self.fc1(x3))
        x3 = self.fc2(x3)
        return x3