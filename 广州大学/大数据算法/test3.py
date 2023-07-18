import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.cluster import KMeans, MeanShift, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys
os.chdir(sys.path[0])
# 数据预处理和特征提取
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor()  # 转换为张量
])

dataset = datasets.ImageFolder('GTSRB/train', transform=transform)

# # 选择部分数据集进行实验
# subset_indices = torch.randperm(len(dataset))[:500]  # 选择1000个样本
# subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

# 选择数据集的前10个类
selected_classes = range(5)
subset_indices = []
for class_idx in selected_classes:
    class_indices = torch.where(torch.tensor(dataset.targets) == class_idx)[0]
    subset_indices.extend(class_indices[:30])
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    extracted_features = extract_features(images)
    features.append(extracted_features)
    labels.append(_.numpy())

features = torch.cat(features, dim=0).cpu()  # 将特征移回CPU
labels = torch.cat([torch.from_numpy(label) for label in labels], dim=0)  # 将标签转换为Tensor类型

# 过滤出现在标签中的簇
valid_labels = torch.unique(labels)

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=len(valid_labels)) 
kmeans_predicted_labels = kmeans.fit_predict(features)

# 使用Mean Shift进行聚类
mean_shift = MeanShift(bandwidth=42)
mean_shift_predicted_labels = mean_shift.fit_predict(features)

# 使用谱聚类进行聚类
spectral_clustering = SpectralClustering(n_clusters=len(valid_labels), eigen_solver='arpack', affinity='nearest_neighbors')
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
    img_indices = torch.where(kmeans_predicted_labels == i)[0]
    if len(img_indices) > 0:
        img_idx = img_indices[0].item()
        image, _ = subset_dataset[img_idx]
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title("Cluster: {}".format(kmeans_predicted_labels[img_idx]))
    else:
        ax.axis('off')  # 如果簇为空，则不显示图像

for i, ax in enumerate(axes[5:10]):
    img_indices = torch.where(mean_shift_predicted_labels == i)[0]
    if len(img_indices) > 0:
        img_idx = img_indices[0].item()
        image, _ = subset_dataset[img_idx]
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title("Cluster: {}".format(mean_shift_predicted_labels[img_idx]))
    else:
        ax.axis('off')  # 如果簇为空，则不显示图像

for i, ax in enumerate(axes[10:15]):
    img_indices = torch.where(spectral_clustering_predicted_labels == i)[0]
    if len(img_indices) > 0:
        img_idx = img_indices[0].item()
        image, _ = subset_dataset[img_idx]
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title("Cluster: {}".format(spectral_clustering_predicted_labels[img_idx]))
    else:
        ax.axis('off')  # 如果簇为空，则不显示图像

plt.tight_layout()
plt.show()

# 散点图分析

# 提取特征向量
features_np = features.numpy()

# 使用PCA降维至2维
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_np)
# 绘制散点图

plt.figure(figsize=(16, 4))

# 绘制原始数据的散点图
plt.subplot(1, 4, 1)
for label in valid_labels:
    indices = torch.where(labels == label)[0]
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label="Class {}".format(label))
plt.title("Original Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# 绘制K-means聚类的散点图
plt.subplot(1, 4, 2)
for label in torch.unique(kmeans_predicted_labels):
    indices = torch.where(kmeans_predicted_labels == label)[0]
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label="Cluster {}".format(label))
plt.title("K-means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# 绘制Mean Shift聚类的散点图
plt.subplot(1, 4, 3)
for label in torch.unique(mean_shift_predicted_labels):
    indices = torch.where(mean_shift_predicted_labels == label)[0]
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label="Cluster {}".format(label))
plt.title("Mean Shift Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# 绘制Spectral Clustering聚类的散点图
plt.subplot(1, 4, 4)
for label in torch.unique(spectral_clustering_predicted_labels):
    indices = torch.where(spectral_clustering_predicted_labels == label)[0]
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label="Cluster {}".format(label))
plt.title("Spectral Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.tight_layout()
plt.show()

