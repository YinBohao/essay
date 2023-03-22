import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# 输入数据集大小和维度，生成随机数据集
size = 100
dimension = 2

# 设置超参数
learning_rate = 1
num_epochs = 20

# 设置随机种子，保证结果可重现
# torch.manual_seed(0)
# random.seed(0)

# 随机生成数据集


def generate_dataset(num_samples, input_size):
    # 随机生成输入张量
    inputs = torch.randn(num_samples, input_size)

    # 生成随机标签1,-1
    labels = torch.where(torch.matmul(inputs, torch.randn(input_size, 1)) > 0,
                         torch.ones(num_samples, 1),
                         (torch.zeros(num_samples, 1)-1))

    return inputs, labels

# 训练，自定义迭代次数与学习率


def train_perceptron(inputs, labels, num_epochs, learning_rate):
    # 初始化权重与偏置
    w = torch.zeros(inputs.size(1), 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for epoch in range(num_epochs):
        # 在每个样本上进行训练
        for input, label in zip(inputs, labels):

            # 计算模型输出
            output = torch.matmul(input, w) + b
            prediction = torch.where(
                output > 0, torch.ones(1), (torch.zeros(1)-1))

            # 计算损失
            loss = torch.max(torch.tensor(0.), -label * output)

            # 计算梯度并更新参数
            loss.backward()
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
                w.grad.zero_()
                b.grad.zero_()

        # 打印当前准确度
        accuracy = compute_accuracy(inputs, labels, w, b)
        print(f"Epoch {epoch}: Accuracy = {accuracy:.2f}")

    # print(w,b)
    return w, b

# 计算准确率


def compute_accuracy(inputs, labels, w, b):
    with torch.no_grad():
        output = torch.matmul(inputs, w) + b
        prediction = torch.where(output > 0, torch.ones(1), (torch.zeros(1)-1))
        accuracy = (prediction == labels).sum().item() / len(labels)
    return accuracy


# 读取数据与标签
inputs, labels = generate_dataset(size, dimension)

# 训练感知机模型
w, b = train_perceptron(inputs, labels, num_epochs, learning_rate)
print(w)
print(b)

with torch.no_grad():
    y_pred = torch.matmul(inputs, w) + b
    y_pred = torch.sign(y_pred).reshape(-1)

plt.title('Perceptron Learning Algorithm', size=14)
plt.xlabel('$x^{(0)}$', size=14)
plt.ylabel('$x^{(1)}$', size=14)

# 绘制分离线
xData = np.linspace(min(inputs[:,0]), max(inputs[:,0]), 100)
yData = (b.detach().numpy() - w[0].detach().numpy() * xData) / w[1].detach().numpy()
plt.plot(xData, yData, color='b')

# 绘制数据点
plt.scatter(inputs[:, 0], inputs[:, 1], c=y_pred)
plt.show()
