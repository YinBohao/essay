import sys, os
os.chdir(sys.path[0])

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([
    #图像增强
    # transforms.Resize(120),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(96),
    # transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5),
    #转变为tensor 正则化
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #正则化
])

trainset=torchvision.datasets.CIFAR10(
    root=r'data',
    train=True,
    # download=True,
    transform=transform
)
# 加载训练集, shuffle参数指定了是否对数据进行随机排序
# 设置shuffle=True来打乱数据集，以便每个批次中包含的图像是随机的
trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True, #乱序
    # num_workers=4,
)

testset=torchvision.datasets.CIFAR10(
    root=r'data',
    train=False,
    # download=True,
    transform=transform
)
testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    # num_workers=2
)
import matplotlib.pyplot as plt
import numpy as np

# 获取第一张图像
images, labels = next(iter(trainloader))
image = images[120]

# 将 PyTorch 张量转换为 NumPy 数组
image_np = image.numpy()

# 将数组中的图像通道维度调整到最后一个维度
image_np = np.transpose(image_np, (1, 2, 0))

# print(image_np)
# print(image_np2)
# 显示图像
plt.imshow(image_np)
plt.show()

# for i, data in enumerate(trainloader):
# 	# i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
#     print("第 {} 个Batch \n{}".format(i, len(data)))
"""
基于Pytorch和学过的卷积神经网络模型及优化算法实现对CIFAR-10 图片集的识别,
至少用3种方法实现,比较各种方法的运行时间和识别准确率。
要求说明每种方法主要借鉴了哪种已知的深度学习模型.
加分项：实现方法越多得分越高；优化方法运用越多得分越高；
文档的可读性强、表达清晰是加分项；
模型或优化算法效果无论好坏，能解释清楚原因也是加分项。
"""
"""
基于pytorch和卷积神经网络模型运用不同的优化技巧实现对CIFAR-10数据集的识别，
比较各种方法的运行时间和识别准确率，并分析解释

学习率衰减技巧
在训练过程中，可以通过降低学习率来提高模型的精度和稳定性。常见的学习率衰减方法有StepLR、ReduceLROnPlateau、CosineAnnealingLR等。例如，可以在每个epoch结束时，将学习率按照一定的比例进行衰减。
学习率调度（learning rate scheduling）：在训练过程中，将学习率动态地调整，可以提高模型的收敛速度和准确率。常见的学习率调度方法包括StepLR、ReduceLROnPlateau、CosineAnnealingLR等。


动量优化器
动量优化器可以加速梯度下降的过程，减少收敛时间。常见的动量优化器有SGD、Adam、Adagrad等。其中Adam优化器在实践中表现较好。

数据增强技巧
数据增强技巧可以扩充数据集，提高模型的泛化能力。常见的数据增强方法有随机翻转、随机裁剪、随机旋转、随机亮度变化等。可以使用torchvision.transforms库中的函数进行实现。
通过对训练数据进行一系列变换（例如旋转、翻转、缩放等），可以扩展数据集的大小，提高模型的泛化能力。在PyTorch中，可以使用torchvision.transforms模块实现数据增强。


正则化技巧
正则化技巧可以防止过拟合，提高模型的泛化能力。常见的正则化方法有L1正则化、L2正则化、Dropout等。其中Dropout方法在实践中表现较好。
Dropout正则化：在网络的训练中，我们可以随机的将一部分神经元进行丢弃，以防止过拟合，提高泛化性能。

批标准化（batch normalization）：通过对每个批次的数据进行标准化，可以加速模型的收敛速度，并且可以提高模型的泛化能力。
"""
