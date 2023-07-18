import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

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

# 使用PCA进行降维
pca = PCA(n_components=2)
img_features = pca.fit_transform(img_vector)
# img_features = img_vector

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=len(selected_classes), random_state=400, max_iter=300)
kmeans_labels = kmeans.fit_predict(img_features)

# 使用Mean Shift进行聚类
meanshift = MeanShift(bandwidth=15)
meanshift_labels = meanshift.fit_predict(img_features)

# 可视化原始数据点
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(img_features[:, 0], img_features[:, 1], c=labels)
plt.title('Original Data Points')
# plt.show()

# 可视化K-means聚类结果
plt.subplot(122)
plt.scatter(img_features[:, 0], img_features[:, 1], c=kmeans_labels)
plt.title('K-means Clustering')
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(img_features[:, 0], img_features[:, 1], c=labels)
plt.title('Original Data Points')
# plt.show()
# 可视化Mean Shift聚类结果
plt.subplot(122)
plt.scatter(img_features[:, 0], img_features[:, 1], c=meanshift_labels)
plt.title('Mean Shift Clustering')
plt.show()

from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score

# 计算K-means和Mean Shift的轮廓系数
silhouette_kmeans = silhouette_score(img_features, kmeans_labels)
silhouette_meanshift = silhouette_score(img_features, meanshift_labels)

# 计算调整兰德指数
ari_kmeans = adjusted_rand_score(labels, kmeans_labels)
ari_meanshift = adjusted_rand_score(labels, meanshift_labels)

# 计算正确率
accuracy_kmeans = accuracy_score(labels, kmeans_labels)
accuracy_meanshift = accuracy_score(labels, meanshift_labels)

print("K-means Silhouette Score:", silhouette_kmeans)
print("Mean Shift Silhouette Score:", silhouette_meanshift)
print('ari_kmeans = {}'.format(ari_kmeans) )
print('ari_meanshift = {}'.format(ari_meanshift) )
print('accuracy_kmeans = {}'.format(accuracy_kmeans) )
print('accuracy_meanshift = {}'.format(accuracy_meanshift) )
