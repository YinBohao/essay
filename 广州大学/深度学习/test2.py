import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import matplotlib.pyplot as plt

import sys
import os
os.chdir(sys.path[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_dataset(train_batch_size, test_batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 水平翻转
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

    # train_dataset.data = train_dataset.data[:10000]
    # train_dataset.targets = train_dataset.targets[:10000]

    # test_dataset.data = train_dataset.data[:2000]
    # test_dataset.targets = train_dataset.targets[:2000]

    # 加载训练集, shuffle参数指定了是否对数据进行随机排序
    # 设置shuffle=True来打乱数据集，以便每个批次中包含的图像是随机的
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def train(model, optimizer, criterion, train_loader, device):
    model.train()  # 开启训练模式
    start_time = time.time()

    # 训练一个epoch
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    end_time = time.time()
    epoch_time = end_time - start_time

    return train_loss, train_acc, epoch_time


def test(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total

    return test_loss, test_acc


def train_epochs(train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers, dynamic_lr):
    train_loader, test_loader = load_dataset(
        train_batch_size, test_batch_size, num_workers)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        # 动态修改参数学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= dynamic_lr

        train_loss, train_acc, epoch_time = train(
            model, optimizer, criterion, train_loader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc/100.)

        test_loss, test_acc = test(model, criterion, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc/100.)
        print('Epoch [{:02d}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}%, Time: {:.2f}s, Test Loss: {:.4f}, Test Acc: {:.4f}%'
              .format(epoch + 1, num_epochs, train_loss, train_acc, epoch_time, test_loss, test_acc))

    return train_losses, train_accs, test_losses, test_accs


def train_model(train_batch_size, test_batch_size, num_workers, num_epochs, lr, momentum, optimizer_class, optimizer_name, model_class, model_name, dynamic_lr):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_class.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)
    
    print('{}_{}: '.format(model_name, optimizer_name))
    train_losses, train_accs, test_losses, test_accs = train_epochs(
        train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers, dynamic_lr)

    return train_losses, train_accs, test_losses, test_accs

def plot_results(results, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # fig, ax1 = plt.subplots(figsize=(10, 4))
    # ax2 = ax1.twinx()
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    markers = ['-', '--', '-.', ':']

    for i, result in enumerate(results):
        train_losses, train_accs, test_losses, test_accs, optimizer_name = result

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        x = list(range(len(train_losses)))  # x 轴刻度为每个 epoch 的索引值
        
        ax1.plot(x, train_losses, color=color, linestyle=marker, label=f'Train Loss ({optimizer_name})')
        ax1.plot(x, test_losses, color=color, linestyle=marker, label=f'Test Loss ({optimizer_name})')
        ax2.plot(x, train_accs, color=color, linestyle=marker, label=f'Train Acc ({optimizer_name})')
        ax2.plot(x, test_accs, color=color, linestyle=marker, label=f'Test Acc ({optimizer_name})')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')
    ax2.legend(loc='lower right')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_batch_size = 128
    test_batch_size = 128
    num_workers = 2
    num_epochs = 3
    momentum = 0.9
    models = {
        'LeNet5': LeNet5(),
        'AlexNet': AlexNet(),
        'VGG16': VGG16(),
        'ResNet18': ResNet18()
    }
    
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'Adagrad': torch.optim.Adagrad,
        'RMSprop': torch.optim.RMSprop
    }
    
    learning_rates = {
        'LeNet5': 0.001,
        'AlexNet': 0.001,
        'VGG16': 0.001,
        'ResNet18': 0.001
    }
    results = []
    for model_name, model_class in models.items():
        for optimizer_name, optimizer_class in optimizers.items():
            train_losses, train_accs, test_losses, test_accs = train_model(
                train_batch_size, test_batch_size, num_workers, num_epochs, learning_rates[model_name], momentum, 
                optimizer_class, optimizer_name, model_class, model_name, dynamic_lr=1.0)
            results.append((train_losses, train_accs, test_losses, test_accs, optimizer_name))
        plot_results(results, '{} Optimizer Comparison'.format(model_name))


