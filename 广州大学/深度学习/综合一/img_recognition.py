import sys
import os
os.chdir(sys.path[0])

import random
# 设置随机种子，保证结果可重现
# torch.manual_seed(0)
# random.seed(0)

import optimizer
from VGG16 import VGG16
from ResNet18 import ResNet18
from two_layer_net import TwoLayerNet
from draw import plot_results
from train_test import *
import torch.nn as nn

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  # 使用Xavier均匀初始化
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)  # 将偏置初始化为0

if __name__ == '__main__':
    
    train_batch_size = 128
    test_batch_size = 128
    num_workers = 2
    num_epochs = 20
    momentum = 0.9
    models = {
        # 'TwoLayerNet': TwoLayerNet(),
        'VGG16': VGG16(),
        # 'ResNet18': ResNet18()
    }  
    # VGG
    learning_rates = {
        'SGD': 0.004,
        'MomentumSGD': 0.003,
        'Adam': 0.00001,
        'Adagrad': 0.00005,
        # 'SGD2': 0.001,
        # 'RMSprop': 0.001
    }
    # TWO
    # learning_rates = {
    #     'SGD': 0.003,
    #     'MomentumSGD': 0.003,
    #     'Adam': 0.0003,
    #     'Adagrad': 0.0003,
    #     # 'SGD2': 0.001,
    #     # 'RMSprop': 0.001
    # }
    optimizers = {
        'SGD': optimizer.SGD,
        'MomentumSGD': optimizer.MomentumSGD,
        'Adam': optimizer.Adam,
        'Adagrad': optimizer.AdaGrad,
        # 'SGD2': optimizer.SGD,
        # 'RMSprop': optimizer.RMSprop
    }
    results = []
    for model_name, model_class in models.items():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for optimizer_name, optimizer_class in optimizers.items():
            """
            在每次迭代中，我们创建了一个独立的模型实例，
            并加载了原始模型的参数。这样做是为了确保每个优化器的训练过程是相互独立的，互不影响。
            """
            model_instance = model_class.to(device)
            model_instance.apply(initialize_weights)  # 初始化模型参数
            train_losses, train_accs, test_losses, test_accs = train_model(
                train_batch_size, test_batch_size, num_workers, num_epochs, learning_rates[optimizer_name], momentum, 
                optimizer_class, optimizer_name, model_instance, model_class, model_name, dynamic_lr=1.0)
            results.append((train_losses, train_accs, test_losses, test_accs, optimizer_name))
        plot_results(results, '{} Optimizer Comparison'.format(model_name))

    """
    我们可以观察到，在CIFAR-10数据集上，三种模型都可以取得不错的效果。
    其中，VGG和DenseNet的训练时间稍长，但是最终的准确率相对更高。这也说明了模型的深度和宽度对模型性能的影响。
    在优化算法方面，Adam和RMSprop的表现相对更好，但是在不同的任务上，不同的优化算法可能表现不同，需要根据具体情况进行选择。
    """
    
    # test(vgg, test_loader, device)
