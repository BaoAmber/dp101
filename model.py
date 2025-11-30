import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """简单的全连接神经网络模型用于 MNIST 数字识别"""
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 输入层：28*28 = 784 像素
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        # 输出层：10 个数字类别 (0-9)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # 展平输入图像
        x = x.view(-1, 28 * 28)
        # 第一层 + ReLU 激活
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 第二层 + ReLU 激活
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # 输出层
        x = self.fc3(x)
        return x

