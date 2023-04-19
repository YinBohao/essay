import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import numpy as np

import os, sys
os.chdir(sys.path[0])

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

batch_size = 128
learning_rate = 0.03
num_epochs = 20

# 加载数据集
# 定义数据变换, 数据归一化处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])

# 加载训练集, shuffle参数指定了是否对数据进行随机排序
# 设置shuffle=True来打乱数据集，以便每个批次中包含的图像是随机的
train_set = datasets.MNIST('data', train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)

# 加载测试集
test_set = datasets.MNIST('data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False)

# 多层感知机模型
class MLPModel(nn.Module):
    def __init__(self, activation_func):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        # 根据传入参数选择激活函数
        if activation_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_func == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        out = x.view(-1, self.fc1.in_features)  # 将输入图片展开为向量
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out

# 实例化sigmoid激活模型
model_sigmoid = MLPModel('sigmoid').to(device)

# 实例化relu激活模型
model_relu = MLPModel('relu').to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer_sigmoid = optim.SGD(model_sigmoid.parameters(), lr=learning_rate)
optimizer_relu = optim.SGD(model_relu.parameters(), lr=learning_rate)

# 定义记录列表
sigmoid_eval_acces = []  # 记录测试集准确率
relu_eval_acces = []  # 记录测试集准确率

# sigmoid激活模型训练测试
start_time = time.time()
for epoch in range(num_epochs):
    model_sigmoid.train()  # 开启训练模式
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 移动到GPU计算

        optimizer_sigmoid.zero_grad()  # 梯度清零
        outputs = model_sigmoid(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer_sigmoid.step()  # 更新参数

    model_sigmoid.eval()  # 开启测试模式
    if (epoch+1) % 1 == 0:
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 移动到GPU计算

            outputs = model_sigmoid(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        sigmoid_eval_acces.append(correct/total)
        print('SIGMOID_Epoch [{}/{}], Test Accuracy: {:.2f}%, Time: {:.2f}s'
              .format(epoch+1, num_epochs, 100*correct/total, time.time()-start_time))

# relu激活模型训练测试
start_time = time.time()
for epoch in range(num_epochs):
    model_relu.train()  # 开启训练模式
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 移动到GPU计算

        optimizer_relu.zero_grad()
        outputs = model_relu(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_relu.step()

    model_relu.eval()  # 开启测试模式
    if (epoch+1) % 1 == 0:
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 移动到GPU计算

            outputs = model_relu(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        relu_eval_acces.append(correct/total)
        print('RELU_Epoch [{}/{}], Test Accuracy: {:.2f}%, Time: {:.2f}s'
              .format(epoch+1, num_epochs, 100*correct/total, time.time()-start_time))

plt.title("Test Accuracy")
plt.plot(sigmoid_eval_acces, color='g', label='sigmoid')
plt.plot(relu_eval_acces, color='y', label='relu')
plt.xticks(np.arange(1,21,2))
plt.legend()
plt.show()