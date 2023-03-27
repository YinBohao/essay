import os
import sys
os.chdir(sys.path[0])

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 载入手写数字数据集
train_data = datasets.MNIST(root='data', train=True, 
                            transform=transforms.ToTensor())

test_data = datasets.MNIST(root='data', train=False, 
                           transform=transforms.ToTensor())

# 标准化数据
train_data.data = train_data.data.float() / 255
test_data.data = test_data.data.float() / 255

# 批量加载数据

# 设置超参数
learning_rate = 0.1
batch_size = 256
epochs = 100

# 创建数据加载器, shuffle参数指定了是否对数据进行随机排序
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

# 创建测试数据的数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size, 
                                          shuffle=False)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

class SoftmaxRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        # 调用父类的init
        super().__init__()
        
        # self.linear = torch.nn.Linear(input_dim, output_dim)
        self.w = torch.zeros(size=(input_dim, output_dim), requires_grad=True)
        self.b = torch.zeros(output_dim, requires_grad=True)
        
    def forward(self, x):
        # x = x.view(-1, self.linear.in_features)
        x = x.view(-1, self.w.shape[0]) # 将输入图片展开为向量
        
        # output = self.linear(x) 
        output = torch.matmul(x, self.w) + self.b
        
        # prediction = torch.nn.functional.log_softmax(output, dim=1) 
        prediction = softmax(x) # 使用softmax函数计算概率值
        return prediction


# 创建模型实例
model = SoftmaxRegression(input_dim=28 * 28, output_dim=10)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



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
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))


# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # 计算模型预测值
        outputs = model(images)
        
        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)
        
        # 计算准确率
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('测试准确率: {:.2f}%'.format(100 * correct / total))
