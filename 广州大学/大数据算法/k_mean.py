import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.cluster import KMeans, MeanShift, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

import os
import sys
os.chdir(sys.path[0])

# 数据预处理和特征提取
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor() # 转换为张量
])

dataset = datasets.ImageFolder('GTSRB/train', transform=transform)

# 选择部分数据集进行实验
subset_indices = torch.randperm(len(dataset))[:200]  # 选择1000个样本
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用预训练的ResNet模型提取特征
def extract_features(images):
    resnet = models.resnet50(pretrained=True).to(device)
    resnet.eval()

    with torch.no_grad():
        images = images.to(device)
        features = resnet(images)

    return features.cpu()

features = []
labels = []

for images, _ in dataloader:
    images = images.to(device)  # 将图像移动到GPU上
    # 提取特征（这里假设你有一个预训练的模型来提取特征）
    # 这里的特征提取过程需要根据你的实际情况来完成
    extracted_features = extract_features(images)  
    features.append(extracted_features)
    labels.append(_.numpy())

features = torch.cat(features, dim=0).cpu()  # 将特征移回CPU
labels = torch.cat([torch.from_numpy(label) for label in labels], dim=0)  # 将标签转换为Tensor类型


# 使用K-means进行聚类
kmeans = KMeans(n_clusters=10)  # 假设我们将数据聚类为10个簇
kmeans_predicted_labels = kmeans.fit_predict(features)

# 使用Mean Shift进行聚类
mean_shift = MeanShift()
mean_shift_predicted_labels = mean_shift.fit_predict(features)

# 使用谱聚类进行聚类
spectral_clustering = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='nearest_neighbors')
spectral_clustering_predicted_labels = spectral_clustering.fit_predict(features)

# 定量评估
kmeans_silhouette_avg = silhouette_score(features, kmeans_predicted_labels)
mean_shift_silhouette_avg = silhouette_score(features, mean_shift_predicted_labels)
spectral_clustering_silhouette_avg = silhouette_score(features, spectral_clustering_predicted_labels)

print("K-means Silhouette Score:", kmeans_silhouette_avg)
print("Mean Shift Silhouette Score:", mean_shift_silhouette_avg)
print("Spectral Clustering Silhouette Score:", spectral_clustering_silhouette_avg)

kmeans_predicted_labels = torch.from_numpy(kmeans_predicted_labels)

mean_shift_predicted_labels = torch.from_numpy(mean_shift_predicted_labels)

spectral_clustering_predicted_labels = torch.from_numpy(spectral_clustering_predicted_labels)

# 定性对比分析
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
axes = axes.flatten()

for i, ax in enumerate(axes[:5]):
    img_idx = torch.where(kmeans_predicted_labels == i)[0][0]
    image, _ = subset_dataset[img_idx]
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.set_title("Cluster: {}".format(kmeans_predicted_labels[img_idx]))

for i, ax in enumerate(axes[5:10]):
    img_idx = torch.where(mean_shift_predicted_labels == i)[0][0]
    image, _ = subset_dataset[img_idx]
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.set_title("Cluster: {}".format(mean_shift_predicted_labels[img_idx]))

for i, ax in enumerate(axes[10:15]):
    img_idx = torch.where(spectral_clustering_predicted_labels == i)[0][0]
    image, _ = subset_dataset[img_idx]
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.set_title("Cluster: {}".format(spectral_clustering_predicted_labels[img_idx]))

plt.tight_layout()
plt.show()


# 散点图展示聚类结果
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
scatter1 = plt.scatter(features[:, 0], features[:, 1], c=kmeans_predicted_labels, cmap='Set1', label='K-means')
plt.title("K-means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 3, 2)
scatter2 = plt.scatter(features[:, 0], features[:, 1], c=mean_shift_predicted_labels, cmap='Set1', label='Mean Shift')
plt.title("Mean Shift Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 3, 3)
scatter3 = plt.scatter(features[:, 0], features[:, 1], c=spectral_clustering_predicted_labels, cmap='Set1', label='Spectral Clustering')
plt.title("Spectral Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 创建图例
legend = plt.legend(handles=[scatter1, scatter2, scatter3], loc='best')
plt.setp(legend.get_texts(), color='black')  # 设置图例文本颜色为黑色

plt.tight_layout()
plt.show()