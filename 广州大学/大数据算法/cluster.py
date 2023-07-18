import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

import os
import sys
os.chdir(sys.path[0])

# 设置设备（GPU或CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据集根目录
root = "GTSRB/train"

# 要选择的类别和每个类别的图像数量
selected_classes = ["00000", "00001", "00002", "00003", "00004"]
# , "00005", "00006", "00007", "00008", "00009"
num_images_per_class = 20

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

# 将图像扁平化为二维数组
img_vector = []
labels = []
for img, label in subset_dataset:
    img_vector.append(img.view(-1).numpy())
    labels.append(label)
img_vector = np.array(img_vector)

# 将numpy数组转换为tensor对象，并移动到GPU上
img_tensor = torch.from_numpy(img_vector).to(device)

# 高维聚类视角 - K-means聚类 + PCA降维
pca = PCA(n_components=2)
img_pca = pca.fit_transform(img_tensor.cpu().numpy())

kmeans_pca = KMeans(n_clusters=len(selected_classes))
kmeans_pca.fit(img_pca)
cluster_labels_pca = kmeans_pca.labels_

# 高维聚类视角 - K-means聚类 + t-SNE降维
tsne = TSNE(n_components=2)
img_tsne = tsne.fit_transform(img_tensor.cpu().numpy())

kmeans_tsne = KMeans(n_clusters=len(selected_classes))
kmeans_tsne.fit(img_tsne)
cluster_labels_tsne = kmeans_tsne.labels_

# 可视化聚类结果
# 高维聚类结果（PCA降维）
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(img_pca[:, 0], img_pca[:, 1], c=cluster_labels_pca, cmap='rainbow')
plt.title("Clustering (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# 高维聚类结果（t-SNE降维）
plt.subplot(122)
plt.scatter(img_tsne[:, 0], img_tsne[:, 1], c=cluster_labels_tsne, cmap='rainbow')
plt.title("Clustering (t-SNE)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.tight_layout()
plt.show()
