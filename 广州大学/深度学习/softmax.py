import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import os
import sys
os.chdir(sys.path[0])


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from IPython.display import clear_output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 设置超参数
learning_rate = 0.03
batch_size = 256
epochs = 20

# 加载数据集
# 定义数据变换
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])

# 加载训练集, shuffle参数指定了是否对数据进行随机排序, 设置shuffle=True来打乱数据集，以便每个批次中包含的图像是随机的
train_set = datasets.MNIST('data', train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)

# 加载测试集
test_set = datasets.MNIST('data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False)

# 定义单层神经网络，最后一层使用softmax激活函数，使之可以使用交叉熵损失函数进行训练
class SoftmaxRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()# 调用父类的init
        self.fc1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)# 将输入图片展开为向量
        x = torch.nn.functional.softmax(self.fc1(x), dim=1)
        return x

# 创建模型实例
model = SoftmaxRegression(
    input_dim=28 * 28, output_dim=10).to(device)

# 定义损失函数和优化器, 使用交叉熵损失和随机梯度下降优化器
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(model, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            clear_output(wait=True)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            plt.clf()
            plt.imshow(data[0].cpu().view(28, 28), cmap='gray')
            plt.title(f"Prediction: {output[0].argmax().item()}")
            plt.pause(0.5)



if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)


# # 训练
# losses = []  # 记录训练集损失
# acces = []  # 记录训练集准确率
# eval_losses = []  # 记录测试集损失
# eval_acces = []  # 记录测试集准确率

# for epoch in range(epochs):
#     train_loss = 0
#     train_acc = 0
#     model.train()  # 指明接下来model进行的是训练过程
#     # 动态修改参数学习率
#     # if epoch % 5 == 0:
#     #     optimizer.param_groups[0]['lr'] *= 0.9
#     for img, label in train_loader:
#         img, label = img.to(device), label.to(device)# 移动到GPU计算

#         # 前向传播
#         out = model(img)
#         loss = criterion(out, label)
        
#         # 反向传播
#         optimizer.zero_grad()  # 清空上一轮的梯度
#         loss.backward()  # 根据前向传播得到损失，再由损失反向传播求得各个梯度
#         optimizer.step()  # 根据反向传播得到的梯度优化模型中的参数

#         train_loss += loss.item()  # 所有批次损失的和
        
#         # 计算分类的准确率
#         pred=torch.argmax(out,1)
#         num_correct = (pred == label).sum().item()
#         acc = num_correct / img.shape[0]  # 每一批样本的准确率
#         train_acc += acc

#     losses.append(train_loss / len(train_loader))  # 所有样本平均损失
#     acces.append(train_acc / len(train_loader))  # 所有样本的准确率

#     # 运用训练好的模型在测试集上检验效果
#     eval_loss = 0
#     eval_acc = 0
#     model.eval()  # 指明接下来要进行模型测试（不需要反向传播）
#     # with torch.no_grad():
#     for img, label in test_loader:
#         img, label = img.to(device), label.to(device)
#         out = model(img)
#         loss = criterion(out, label)
        
#         # 记录误差
#         eval_loss += loss.item()
        
#         # 记录准确率
#         pred=torch.argmax(out,1)
#         num_correct = (pred == label).sum().item()
#         acc = num_correct / img.shape[0]
#         eval_acc += acc

#     eval_losses.append(eval_loss / len(test_loader))
#     eval_acces.append(eval_acc / len(test_loader))

#     print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
#           .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
#                   eval_loss / len(test_loader), eval_acc / len(test_loader)))

# plt.title('train loss')
# plt.plot(np.arange(len(losses)), losses)
# plt.legend(['Train Loss'], loc='best')
# plt.show()
