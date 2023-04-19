from PIL import Image
import matplotlib.pyplot as plt
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
            # 正则化技术:包括L1, L2和Dropout等技术。
            # L1和L2正则化可以向损失函数中添加一个与网络权重相关的正则项,使得网络权重越小,从而防止过拟合。
            # Dropout可以在每个训练迭代中随机的将一些神经元设置为0,从而使得拟合程度不会过度依赖某些特定的神经元。
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
# , weight_decay=0.01

num_epochs = 20

# 定义记录列表
train_losses = []  # 记录训练集损失
train_acces = []  # 记录训练集准确率
eval_losses = []  # 记录测试集损失
eval_acces = []  # 记录测试集准确率

for epoch in range(num_epochs):
    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.9

    train_correct, train_total, train_loss = 0, 0, 0
    model.train()  # 开启训练模式
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)  # 移动到GPU计算
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # 损失和

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    test_correct, test_total, test_loss = 0, 0, 0
    model.eval()  # 开启测试模式
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)  # 移动到GPU计算
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()  # 损失和

        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    train_loss = train_loss / train_total
    train_losses.append(train_loss)
    train_accuracy = 100. * train_correct / train_total
    train_acces.append(train_accuracy/100.)
    test_loss = test_loss / test_total
    eval_losses.append(test_loss)
    test_accuracy = 100. * test_correct / test_total
    eval_acces.append(test_accuracy/100.)
    # print('Epoch: {}, Accuracy: {}%'.format(epoch, accuracy))
    print('Epoch: {},train_loss: {:.8f}, train_accuracy: {:.4f}%, test_loss: {:.8f}, test_accuracy: {:.4f}%'.format(
        epoch, train_loss, train_accuracy, test_loss, test_accuracy))
# model.load_state_dict(torch.load('model.pth'))

# 绘制结果
plt.plot(train_losses, color='b', label='train_losses')
plt.plot(train_acces, color='r', label='train_acces')
plt.plot(eval_losses, color='g', label='eval_losses')
plt.plot(eval_acces, color='y', label='eval_acces')
plt.legend()
plt.show()