# import torch

# # 输入数据维度n
# n = int(input('Please enter data dimension n: '))
# # 随机生成n维特征，以及对应数量权值，设置需要求导
# x = torch.randn(1, n)
# w = torch.randn(1, n, requires_grad=True)

# import torch
# w = torch.randn((input_dim, output_dim), requires_grad=True)
# b = torch.zeros((output_dim), requires_grad=True)

# z = torch.matmul(x, w) + b
# y = torch.sigmoid(z)

# loss = -(y_true * torch.log(y) + (1 - y_true) * torch.log(1 - y)).mean()

# loss.backward()

# with torch.no_grad():
#     w -= learning_rate * w.grad
#     b -= learning_rate * b.grad
#     w.grad.zero_()
#     b.grad.zero_()

import torch

# 定义输入数据和标签
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
y = torch.tensor([[-1.0], [1.0], [1.0]])

# 定义模型参数
w = torch.randn(3, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 定义模型
def model(x):
    return torch.matmul(x, w) + b

# 定义损失函数
def loss_fn(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

# 定义优化器
learning_rate = 0.01
optimizer = torch.optim.SGD([w, b], lr=learning_rate)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    y_pred[y_pred>=0] = 1
    y_pred[y_pred<0] = -1
    accuracy = torch.mean((y_pred == y).float())
    
    # 打印训练信息
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, loss.item(), accuracy.item()))
