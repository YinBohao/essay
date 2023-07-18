import sys
import os
os.chdir(sys.path[0])
import torch
import torch.nn as nn

# 定义ResNet-18模型

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 批标准化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # 批标准化层
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 步长
        self.stride = stride

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 如果步长不为1或者输入通道数与输出通道数不一致，使用1x1卷积调整通道数
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差连接
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # 批标准化层
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第一组残差块
        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        # 第二组残差块
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        # 第三组残差块
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        # 第四组残差块
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)
        # 自适应平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        # 添加第一个残差块
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # 添加剩余的残差块
        for i in range(num_blocks-1):
            layers.append(block(self.in_channels, out_channels))
        # 将所有的残差块打包成Sequential容器
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 自适应平均池化
        out = self.avg_pool(out)
        # 展平
        out = out.view(out.size(0), -1)
        # 全连接层
        out = self.fc(out)
        return out
