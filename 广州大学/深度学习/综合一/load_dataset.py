import sys
import os
os.chdir(sys.path[0])

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_dataset(train_batch_size, test_batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 水平翻转
        # 转变为tensor 正则化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正则化
    ])

    transform_test = transforms.Compose([
        # 转变为tensor 正则化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正则化
    ])

    """
    transform_train 是针对训练集的数据增强操作，包括随机裁剪、随机旋转、水平翻转等；
    transform_test 是针对测试集的操作，只进行标准化处理。
    """

    train_dataset = datasets.CIFAR10(
        root=r'data', train=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root=r'data', train=False, transform=transform_test)
    
    # 选取部分训练集和测试集测试，以减少训练时间
    # train_dataset.data = train_dataset.data[:10000]
    # train_dataset.targets = train_dataset.targets[:10000]

    # test_dataset.data = train_dataset.data[:2000]
    # test_dataset.targets = train_dataset.targets[:2000]

    # 加载训练集, shuffle参数指定了是否对数据进行随机排序
    # 设置shuffle=True来打乱训练集，以便每个批次中包含的图像是随机的
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader