import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import os
import sys
os.chdir(sys.path[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 设置超参数
learning_rate = 0.01
batch_size = 128
epochs = 20

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

# 定义神经网络，最后一层使用softmax激活函数，使之可以使用交叉熵损失函数进行训练


class SoftmaxRegression(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super().__init__()  # 调用父类的init, python3.X后可省略括号内容
        self.fc1 = torch.nn.Linear(input_dim, output_dim)
        self.fc2 = torch.nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)  # 将输入图片展开为向量
        x = torch.nn.functional.relu(self.fc1(x))  # 隐藏层使用relu激活
        x = torch.nn.functional.log_softmax(self.fc2(x), dim=1)
        return x


# 创建模型实例，转入GPU计算
model = SoftmaxRegression(
    input_dim=28 * 28, hid_dim=256, output_dim=10).to(device)

# 定义损失函数和优化器, 使用交叉熵损失和随机梯度下降优化器
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 定义记录列表
losses = []  # 记录训练集损失
acces = []  # 记录训练集准确率
eval_losses = []  # 记录测试集损失
eval_acces = []  # 记录测试集准确率

for epoch in range(epochs):
    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.9
    # 训练
    train_loss = 0
    train_acc = 0
    model.train()  # 开启训练模式
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)  # 移动到GPU计算

        out = model(img)  # 前向传播
        loss = criterion(out, label)  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item()  # 损失和

        # 计算训练集准确率
        pred = torch.argmax(out, 1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / batch_size
        train_acc += acc

    losses.append(train_loss / len(train_loader))  # 所有样本平均损失
    acces.append(train_acc / len(train_loader))  # 所有样本的准确率

    # 测试
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 开启测试模式
    # with torch.no_grad():  # 不计算梯度
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        out = model(img)
        loss = criterion(out, label)

        eval_loss += loss.item()

        pred = torch.argmax(out, 1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / batch_size
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))

    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                  eval_loss / len(test_loader), eval_acc / len(test_loader)))
# 绘制结果
plt.plot(np.arange(epochs), losses, color='b', label='train_losses')
plt.plot(np.arange(epochs), acces, color='r', label='train_acces')
plt.plot(np.arange(epochs), eval_acces, color='g', label='eval_acces')
plt.legend()
plt.show()
