import sys
import os
os.chdir(sys.path[0])
import torch
import torch.nn as nn

# 定义VGG-16模型

import torch
import torch.nn as nn

class VGG16(nn.Module):
    """ 
        网络结构如下所示
        conv - batchnorm - relu - conv - batchnorm - relu - maxpool -
        conv - batchnorm - relu - conv - batchnorm - relu - maxpool -
        conv - batchnorm - relu - conv - batchnorm - relu - conv - batchnorm - relu -maxpool -
        conv - batchnorm - relu - conv - batchnorm - relu - conv - batchnorm - relu -maxpool -
        conv - batchnorm - relu - conv - batchnorm - relu - conv - batchnorm - relu -maxpool -
        avgpool - flatten - 
        affine - relu - dropout - affine - relu - dropout - affine
    """
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        # 定义卷积层和归一化层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入通道数为3，输出通道数为64，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(64),  # 归一化层，对每个通道进行归一化
            nn.ReLU(inplace=True),  # ReLU激活函数，inplace=True表示将结果直接覆盖到输入变量上，节省内存
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 输入通道数为64，输出通道数为64，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层，池化核大小为2x2，步长为2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输入通道数为64，输出通道数为128，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 输入通道数为128，输出通道数为128，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 输入通道数为128，输出通道数为256，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 输入通道数为256，输出通道数为256，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 输入通道数为256，输出通道数为256，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 输入通道数为256，输出通道数为512，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 输入通道数为512，输出通道数为512，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 输入通道数为512，输出通道数为512，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 输入通道数为512，输出通道数为512，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 输入通道数为512，输出通道数为512，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 输入通道数为512，输出通道数为512，卷积核大小为3x3，边界填充为1
            nn.BatchNorm2d(512),  
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )

        # 定义自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 定义分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 输入维度为512*7*7，输出维度为4096
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout(),  # 随机失活层，用于防止过拟合
            nn.Linear(4096, 4096),  # 输入维度为4096，输出维度为4096
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout(),  # 随机失活层
            nn.Linear(4096, num_classes),  # 输入维度为4096，输出维度为num_classes（分类类别数）
        )

    def forward(self, x):
        x = self.features(x)  # 前向传播过程中，将输入数据通过卷积层和归一化层进行特征提取
        x = self.avgpool(x)  # 将特征图进行自适应平均池化
        x = torch.flatten(x, 1)  # 将特征图展平为一维向量
        x = self.classifier(x)  # 将展平后的特征向量输入分类器（全连接层）进行分类
        return x
