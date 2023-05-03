import numpy as np
import random
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import os
import sys
os.chdir(sys.path[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据集根目录
root = "GTSRB/train"

# 要选择的类别和每个类别的图像数量
selected_classes = ["00000", "00001", "00002", "00003", "00004", "00005", "00006", "00007", "00008", "00009"]
num_images_per_class = 200

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载数据集
dataset = ImageFolder(root=root, transform=transform)

# 获取子集索引
subset_indices = []
for class_idx in range(len(dataset.classes)):
    class_name = dataset.classes[class_idx]
    if class_name in selected_classes:
        # 获取该类别的所有图像索引
        indices = [idx for idx in range(len(dataset.targets)) if dataset.targets[idx] == class_idx]
        # 随机选择指定数量的图像索引
        subset_indices.extend(random.sample(indices, num_images_per_class))

# 获取子集数据集
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)   

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 将图像扁平化为二维数组
img_vector = []
labels = []
for img, label in subset_dataset:
    img_vector.append(img.view(-1).numpy())
    labels.append(label)
img_vector = np.array(img_vector)

# 将numpy数组转换为tensor对象，并移动到GPU上
img_tensor = torch.from_numpy(img_vector).to(device)


# 对数据进行PCA降维
n_components_2d = 2
pca_model = PCA(n_components=n_components_2d)
img_pca = pca_model.fit_transform(img_tensor.cpu().numpy())

# 对数据进行t-SNE降维
tsne_model = TSNE(n_components=n_components_2d, 
                  perplexity=30, learning_rate=200, 
                  n_iter=1000, random_state=42)
img_tsne = tsne_model.fit_transform(img_tensor.cpu().numpy())

# 对数据进行PCA降维
n_components_3d = 3
pca_model = PCA(n_components=n_components_3d)
img_pca_3d = pca_model.fit_transform(img_tensor.cpu().numpy())

# 对数据进行t-SNE降维
tsne_model = TSNE(n_components=n_components_3d, 
                  perplexity=30, learning_rate=200, 
                  n_iter=1000, random_state=42)
img_tsne_3d = tsne_model.fit_transform(img_tensor.cpu().numpy())

import matplotlib.pyplot as plt

# 将降维结果可视化为2D图形
def plot_2d(X, y, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    targets = np.unique(y)
    for target in targets:
        indices = np.where(y == target)
        ax.scatter(X[indices, 0], X[indices, 1], label=target)
    ax.legend()
    plt.show()

# 将降维结果可视化为3D图形
def plot_3d(X, y, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(title)
    targets = np.unique(y)
    for target in targets:
        indices = np.where(y == target)
        ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], label=target)
    ax.legend()
    plt.show()

# 可视化2D降维结果
plot_2d(img_pca, labels, 'PCA (2D)')
plot_2d(img_tsne, labels, 't-SNE (2D)')

# 可视化3D降维结果
plot_3d(img_pca_3d, labels, 'PCA (3D)')
plot_3d(img_tsne_3d, labels, 't-SNE (3D)')

img = img_vector[0]

# 使用不同的PCA精度进行降维，并展示结果
for n_components in [2, 5, 10, 20, 50, 100]:
    pca_model = PCA(n_components=n_components)
    
    # 使用PCA模型进行降维
    img_pca = pca_model.fit_transform(img.reshape(1, -1))
    img_reconstructed_pca = pca_model.inverse_transform(img_pca)
    
    tsne_model = PCA(n_components=n_components)
    
    # 使用t-SNE模型进行降维
    img_tsne = tsne_model.fit_transform(img.reshape(1, -1))
    img_reconstructed_tsne = tsne_model.inverse_transform(img_tsne)
    
    # 显示原始图像和PCA降维图像
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].imshow(img.reshape(32, 32, 3))
    axes[0].set_title('Original Image')
    axes[1].imshow(img_reconstructed_pca.reshape(32, 32, 3))
    axes[1].set_title('PCA Reconstruction (n_components={})'.format(n_components))
    plt.show()
    
    # 显示原始图像和t-SNE降维图像
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].imshow(img.reshape(32, 32, 3))
    axes[0].set_title('Original Image')
    axes[1].imshow(img_reconstructed_tsne.reshape(32, 32, 3))
    axes[1].set_title('t-SNE Reconstruction (perplexity={}, learning_rate={})'.format(tsne_model.perplexity, tsne_model.learning_rate))
    plt.show()

