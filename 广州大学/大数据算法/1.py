from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import os
import sys
os.chdir(sys.path[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义转换方法
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),  # 中心裁剪
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5,), std=(0.5,)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化处理
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
trainset = datasets.ImageFolder('GTSRB/train', transform=transform)
testset = datasets.ImageFolder('GTSRB/test', transform=transform)

# 加载数据
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# # 打印数据集的类别数和样本数
# print(f'Num of classes: {len(trainset.classes)}, Num of samples: {len(trainset)}')

# # 读取数据集中的一批图像和标签
# for images, labels in trainloader:
#     print(images.shape)  # 输出这批图像的shape
#     print(labels)  # 输出这批图像对应的标签
#     print(len(labels))
#     break

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # 如果输入与输出通道数不一致，需要使用1x1卷积进行变换
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 加入跨层连接
        out += self.shortcut(identity)
        out = self.relu(out)

        return out

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            '''
            正则化技术：包括L1, L2和Dropout等技术。

            L1和L2正则化可以向损失函数中添加一个与网络权重相关的正则项，使得网络权重越小，从而防止过拟合。

            Dropout可以在每个训练迭代中随机的将一些神经元设置为0，从而使得拟合程度不会过度依赖某些特定的神经元。
            ''',
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(512, 43)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc_layer(x)
        return x


# 创建模型实例，转入GPU计算
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, weight_decay=0.01)

num_epochs = 20

for epoch in range(num_epochs):
    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.9

    train_correct, train_total = 0, 0
    model.train()  # 开启训练模式
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)  # 移动到GPU计算
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    test_correct, test_total = 0, 0
    model.eval()  # 开启测试模式
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)  # 移动到GPU计算
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
    train_accuracy = 100. * train_correct / train_total
    test_accuracy = 100. * test_correct / test_total
    # print('Epoch: {}, Accuracy: {}%'.format(epoch, accuracy))
    print('Epoch: {},train_accuracy: {}%, test_accuracy: {}%'.format(
        epoch, train_accuracy, test_accuracy))
# model.load_state_dict(torch.load('model.pth'))

