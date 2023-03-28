from torchvision import datasets, transforms
import torch

import os
import sys
os.chdir(sys.path[0])


# 设置超参数
learning_rate = 0.1
batch_size = 256
epochs = 100

# 加载数据集

# 定义数据变换
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载训练集, shuffle参数指定了是否对数据进行随机排序, 设置shuffle=True来打乱数据集，以便每个批次中包含的图像是随机的
train_set = datasets.MNIST('data', train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)

# 加载测试集
test_set = datasets.MNIST('data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False)

# 定义一个简单的神经网络，最后一层使用softmax激活函数，使之可以使用交叉熵损失函数进行训练
class SoftmaxRegression(torch.nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim):
        # 调用父类的init
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, middle_dim)
        self.fc3 = torch.nn.Linear(middle_dim, output_dim)

    def forward(self, x):
        # 将输入图片展开为向量
        x = x.view(-1, self.fc1.in_features)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)
        return x


# 创建模型实例
device = torch.device('cuda:0')
model = SoftmaxRegression(
    input_dim=28 * 28, middle_dim=256, output_dim=10).to(device)

# 定义损失函数和优化器, 使用交叉熵损失和随机梯度下降优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# def train(model, train_loader, criterion, optimizer, epochs):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))


# 训练模型
for epoch in range(epochs):
    for images, labels in train_loader:
        # 计算模型预测值
        outputs = model(images)

        # 计算损失值
        loss = criterion(outputs, labels)

        # 反向传播及优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每10个epoch输出一次结果
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
              1, epochs, loss.item()))


# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # 计算模型预测值
        outputs = model(images)

        # 获取预测结果
        predicted = torch.max(outputs.data, 1)

        # 计算准确率
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('测试准确率: {:.2f}%'.format(100 * correct / total))
