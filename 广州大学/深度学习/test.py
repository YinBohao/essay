import torch
import numpy as np

import os
import sys
os.chdir(sys.path[0])

#导入 pytorch 内置的mnist数据集
from torchvision.datasets import mnist

#导入对图像的预处理模块
import torchvision.transforms as transforms

#导入dataset的分批读取包
from torch.utils.data import DataLoader

#导入神经网络包nn（可用来定义和运行神经网络）
from torch import nn

#functional这个包中包含了神经网络中使用的一些常用函数，这些函数的特点是:不具有可学习的参数(如ReLU，pool，DropOut等)
import torch.nn.functional as F

#optim中实现了大多数的优化方法来更新网络权重和参数，如SGD、Adam
import torch.optim as optim

#导入可视化绘图库
import matplotlib.pyplot as plt
#2定义代码中用到的各个超参数

train_batch_size = 64  #指定DataLoader在训练集中每批加载的样本数量
test_batch_size = 128  #指定DataLoader在测试集中每批加载的样本数量
num_epoches = 20 # 模型训练轮数
lr = 0.01  #设置SGD中的初始学习率
momentum = 0.5 #设置SGD中的冲量
#3对数据进行预处理
# Compose方法即是将两个操作合并一起
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.1307], [0.3081])])
#4下载和分批加载数据集

#将训练和测试数据集下载到同目录下的data文件夹下
train_dataset = mnist.MNIST('data', train=True, transform=transform)
test_dataset = mnist.MNIST('data', train=False, transform=transform)

#dataloader是一个可迭代对象，可以使用迭代器一样使用。
#其中shuffle参数为是否打乱原有数据顺序
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
#5定义一个神经网络模型

class Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Net,self).__init__()  
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3=nn.Linear(n_hidden_2,out_dim)  #最后一层接Softmax所以不需要ReLU激活
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x
# Sequential() 即相当于把多个模块按顺序封装成一个模块
        
#实例化网络模型

#检测是否有可用的GPU，否则使用cpu
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
#网络模型参数分别为：输入层大小、隐藏层1大小、隐藏层2大小、输出层大小（10分类）
model=Net(28*28,300,100,10)
#将模型移动到GPU加速计算
model.to(device)

#定义模型训练中用到的损失函数和优化器
criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=momentum)
# parameters()将model中可优化的参数传入到SGD中
#6对模型进行训练
# 开始训练
losses = []       #记录训练集损失
acces = []        #记录训练集准确率
eval_losses = []  #记录测试集损失
eval_acces = []  #记录测试集准确率

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()  #指明接下来model进行的是训练过程
    #动态修改参数学习率
    if epoch%5==0:
        optimizer.param_groups[0]['lr'] *= 0.9
    for img, label in train_loader:
        img = img.to(device) #将img移动到GPU计算
        label = label.to(device)
        img = img.view(img.size(0), -1)#把输入图像的维度由四维转化为2维，因为在torch中只能处理二维数据
        #img.size(0)为取size的第0个参数即此批样本的个数，-1为自适应参数
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad() #先清空上一轮的梯度
        loss.backward()  #根据前向传播得到损失，再由损失反向传播求得各个梯度
        optimizer.step() #根据反向传播得到的梯度优化模型中的参数
        
        train_loss += loss.item()  # 所有批次损失的和
        # 计算分类的准确率
        _, pred = out.max(1) #返回输出二维矩阵中每一行的最大值及其下标，1含义为以第1个维度（列）为参考
        #pred=torch.argmax(out,1)
        num_correct = (pred == label).sum().item()
        #num_correct = pred.eq(label).sum().item()
        acc = num_correct / img.shape[0]  #每一批样本的准确率
        train_acc += acc
        
    losses.append(train_loss / len(train_loader)) #所有样本平均损失
    acces.append(train_acc / len(train_loader)) #所有样本的准确率

    # 7运用训练好的模型在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    # 将模型改为预测模式
    model.eval() #指明接下来要进行模型测试（不需要反向传播）
    #with torch.no_grad():
    for img, label in test_loader:
        img=img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), 
                     eval_loss / len(test_loader), eval_acc / len(test_loader)))
#8通过可视化的方法输出模型性能结果
# plt.title('train loss')
# plt.plot(np.arange(len(losses)), losses)
# plt.legend(['Train Loss'], loc='best')
plt.plot(np.arange(len(losses)), losses, color='b',label='train_losses')
plt.plot(np.arange(len(losses)), acces, color='r',label='train_acces')
plt.plot(np.arange(len(losses)), eval_acces, color='g',label='eval_acces')
plt.legend()
plt.show()