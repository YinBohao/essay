import sys
import os
os.chdir(sys.path[0])

import torch
import time
from load_dataset import load_dataset
from LossFunction import *

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

def train_model(train_batch_size, test_batch_size, num_workers, num_epochs, lr, momentum, optimizer_class, optimizer_name, model, model_class, model_name, dynamic_lr):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = model_class.to(device)
    criterion = LossFunction()
    
    if optimizer_name in ['MomentumSGD']:
        optimizer = optimizer_class(model.parameters(), lr, momentum=momentum)
    else:
        optimizer = optimizer_class(model.parameters(), lr)
    
    print('{}_{}: '.format(model_name, optimizer_name))
    train_losses, train_accs, test_losses, test_accs = train_epochs(
        train_batch_size, test_batch_size, num_epochs, model, optimizer, criterion, device, num_workers, dynamic_lr)

    return train_losses, train_accs, test_losses, test_accs