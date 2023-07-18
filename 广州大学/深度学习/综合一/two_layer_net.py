import sys
import os
os.chdir(sys.path[0])

import torch.nn as nn

class TwoLayerNet(nn.Module):
    """ 
        网络结构如下所示
        conv - relu - conv- relu - affine
    """
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        # 定义第一个卷积层，输入通道数为3，输出通道数为16，卷积核大小为3，步长为1，填充为1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        # 定义第二个卷积层，输入通道数为16，输出通道数为32，卷积核大小为3，步长为1，填充为1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 定义全连接层，输入大小为32 * 32 * 32，输出大小为10
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        # 进行第一次卷积操作，然后应用ReLU激活函数
        x = self.relu(self.conv1(x))
        # 进行第二次卷积操作，然后应用ReLU激活函数
        x = self.relu(self.conv2(x))
        # 将特征图展平为向量
        x = x.view(x.size(0), -1)
        # 进行全连接操作
        x = self.fc(x)
        return x



    """ 
        网络结构如下所示
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """


