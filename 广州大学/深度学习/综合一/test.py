import sys
import os
os.chdir(sys.path[0])
import optimizer
from two_layer_net import TwoLayerNet
from draw import plot_results
import torch
import torch.nn as nn
import time
from load_dataset import load_dataset

# 定义网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    # model.eval()
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
        # if epoch % 5 == 0:
        #     optimizer.param_groups[0]['lr'] *= dynamic_lr

        train_loss, train_acc, epoch_time = train(
            model, optimizer, criterion, train_loader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc/100.)

        test_loss, test_acc = test(model, criterion, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc/100.)
        print('Epoch [{:02d}/{:02d}], Train Loss: {:.4f}, Train Acc: {:.4f}%, Time: {:.2f}s, Test Loss: {:.4f}, Test Acc: {:.4f}%'
              .format(epoch + 1, num_epochs, train_loss, train_acc, epoch_time, test_loss, test_acc))

    return train_losses, train_accs, test_losses, test_accs

def train_model(train_batch_size, test_batch_size, num_workers, num_epochs, lr, momentum, optimizer_class, optimizer_name, model_class, model_name, dynamic_lr):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_class.to(device)
    # criterion = cross_entropy_error(）
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Momentum':
        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)
    
    print('{}_{}: '.format(model_name, optimizer_name))
    train_losses, train_accs, test_losses, test_accs = train_epochs(
        train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers, dynamic_lr)

    return train_losses, train_accs, test_losses, test_accs

if __name__ == '__main__':
    
    train_batch_size = 128
    test_batch_size = 128
    num_workers = 2
    num_epochs = 20
    momentum = 0.9
    models = {
        'TwoLayerNet': ConvNet(),
        # 'VGG16': VGG16(),
        # 'ResNet18': ResNet18()
    }  
    learning_rates = {
        'TwoLayerNet': 0.001,
        # 'VGG16': 0.001,
        # 'ResNet18': 0.001
    }
    optimizers = {
        # 'SGD': optimizer.SGD,
        'Momentum': optimizer.MomentumSGD,
        # 'Adam': optimizer.Adam,
        # 'Adagrad': optimizer.AdaGrad,
        # 'RMSprop': optimizer.RMSprop
    }
    results = []
    for model_name, model_class in models.items():
        for optimizer_name, optimizer_class in optimizers.items():
            train_losses, train_accs, test_losses, test_accs = train_model(
                train_batch_size, test_batch_size, num_workers, num_epochs, learning_rates[model_name], momentum, 
                optimizer_class, optimizer_name, model_class, model_name, dynamic_lr=1.0)
            results.append((train_losses, train_accs, test_losses, test_accs, optimizer_name))
        plot_results(results, '{} Optimizer Comparison'.format(model_name))

    """
    我们可以观察到，在CIFAR-10数据集上，三种模型都可以取得不错的效果。
    其中，VGG和DenseNet的训练时间稍长，但是最终的准确率相对更高。这也说明了模型的深度和宽度对模型性能的影响。
    在优化算法方面，Adam和RMSprop的表现相对更好，但是在不同的任务上，不同的优化算法可能表现不同，需要根据具体情况进行选择。
    """
    
    # test(vgg, test_loader, device)
