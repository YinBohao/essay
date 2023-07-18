import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import matplotlib.pyplot as plt

import sys
import os
os.chdir(sys.path[0])
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * correct / total
    train_loss /= len(trainloader)
    return train_acc, train_loss


def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    test_loss /= len(testloader)
    return test_acc, test_loss

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        # 转变为tensor 正则化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正则化
    ])

transform_test = transforms.Compose([
    # 转变为tensor 正则化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正则化
])
"""
transform_train 是针对训练集的数据增强操作，包括随机裁剪、水平翻转等；
transform_test 是针对测试集的数据增强操作，只进行标准化处理。
"""
train_dataset = datasets.CIFAR10(
    root=r'data', train=True, transform=transform_train)
test_dataset = datasets.CIFAR10(
    root=r'data', train=False, transform=transform_test)

train_dataset.data = train_dataset.data[:10000]
train_dataset.targets = train_dataset.targets[:10000]

test_dataset.data = train_dataset.data[:2000]
test_dataset.targets = train_dataset.targets[:2000]

# 加载训练集, shuffle参数指定了是否对数据进行随机排序
# 设置shuffle=True来打乱数据集，以便每个批次中包含的图像是随机的
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False)

# 定义超参数
lr = 0.001
momentum = 0.9
num_epochs = 10
batch_size = 128

# 创建模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Lenet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

for epoch in range(num_epochs):
    train_acc, train_loss = train(net, train_loader, criterion, optimizer, device)
    test_acc, test_loss = test(net, test_loader, criterion, device)
    print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Train Loss={train_loss:.4f}; Test Acc={test_acc:.2f}%, Test Loss={test_loss:.4f}')

test_acc, test_loss = test(net, test_loader, criterion, device)
print(f'Final Test Accuracy = {test_acc:.2f}%, Test Loss = {test_loss:.4f}')

"""

"""

def train_vgg16(train_batch_size, test_batch_size, num_workers, num_epochs, lr, momentum):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG16().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('VGG16: ')
    train_losses, train_accs, test_losses, test_accs = train_epochs(
        train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers)

    return train_losses, train_accs, test_losses, test_accs


def train_resnet18(train_batch_size, test_batch_size, num_workers, num_epochs, lr, momentum):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('ResNet18: ')
    train_losses, train_accs, test_losses, test_accs = train_epochs(
        train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers)

    return train_losses, train_accs, test_losses, test_accs


def train_alexnet(train_batch_size, test_batch_size, num_workers, num_epochs, lr, momentum):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('AlexNet: ')
    train_losses, train_accs, test_losses, test_accs = train_epochs(
        train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers)

    return train_losses, train_accs, test_losses, test_accs


def train_lenet5(train_batch_size, test_batch_size, num_workers, num_epochs, lr, momentum):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('LeNet5: ')
    train_losses, train_accs, test_losses, test_accs = train_epochs(
        train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers)

    return train_losses, train_accs, test_losses, test_accs