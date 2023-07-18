import os
import sys
os.chdir(sys.path[0])

import torch
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

K = 6

# 加载图片
image_path = '00006_00029.png'
# image_path = '00006_00027.png'
image = Image.open(image_path)

# 转换为numpy数组
image_array = np.array(image)

# 将图像数组转换为PyTorch张量
image_tensor = torch.from_numpy(image_array).float()

# 将图像张量重新形状为点集
points = image_tensor.view(-1, 3)

# 使用K均值聚类算法进行聚类
kmeans = KMeans(n_clusters=K)  # K是你希望的聚类数目
kmeans.fit(points)

# 获取每个点的聚类标签
labels = kmeans.labels_

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 假设有一个聚类结果的标签数组labels和对应的数据点集points

# 轮廓系数（Silhouette Coefficient）
silhouette_avg = silhouette_score(points, labels)
print("Average Silhouette Coefficient:", silhouette_avg)

# Calinski-Harabasz指数（Calinski-Harabasz Index）
calinski_harabasz_score = calinski_harabasz_score(points, labels)
print("Calinski-Harabasz Index:", calinski_harabasz_score)

# Dunn指数（Dunn Index）
dunn_score = davies_bouldin_score(points, labels)
print("Dunn Index:", dunn_score)


# 将聚类标签重新形状为与原始图像相同的形状
labels = labels.reshape(image_array.shape[:2])

# 打印每个聚类的像素数量
for i in range(K):
    cluster_size = np.sum(labels == i)
    print(f"Cluster {i+1} size: {cluster_size}")

# 创建一个与原始图像相同大小的矩阵来存储分割结果
segmented_image = np.zeros_like(image_array)

# 为每个聚类设置不同的颜色
colors = kmeans.cluster_centers_ / 255.0  # 将颜色范围归一化到0-1

# 根据聚类标签对像素进行着色
for i in range(K):
    segmented_image[labels == i] = colors[i] * 255 # 将颜色范围还原到0-255

import datetime

now = datetime.datetime.now()
timestamp = now.strftime('%H%M%S')

# 在RGB空间中显示图像分割结果和原始图像对比
fig1 = plt.figure(figsize=(12, 6))

# 左边展示原始图像的点集
ax1 = fig1.add_subplot(121, projection='3d')
r, g, b = image_array[:, :, 0].flatten() / 255.0, image_array[:, :, 1].flatten() / 255.0, image_array[:, :, 2].flatten() / 255.0
ax1.set_title('Original Image')
ax1.scatter(r, g, b, c=image_array.reshape(-1, 3)/255.0, marker='o')
ax1.set_xlabel('Red')
ax1.set_ylabel('Green')
ax1.set_zlabel('Blue')

# 右边展示分割后的点集
ax2 = fig1.add_subplot(122, projection='3d')
ax2.set_title('Segmented Image K={}'.format(K))
ax2.scatter(r, g, b, c=colors[labels.flatten()], marker='o')
ax2.set_xlabel('Red')
ax2.set_ylabel('Green')
ax2.set_zlabel('Blue')

fig1.savefig('figure2/scatter_{}.png'.format(K), dpi=300)
# plt.close()
# plt.show()

# 可视化分割结果
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title('Original Image')
ax1.imshow(image)
ax1.axis('off')

ax2.set_title('Segmented Image K={}'.format(K))
ax2.imshow(segmented_image)
ax2.axis('off')

fig2.savefig('figure2/Image_{}.png'.format(K), dpi=300)
plt.close()
# plt.show()
