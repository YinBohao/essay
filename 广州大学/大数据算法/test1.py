import torch
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datetime

import os
import sys
os.chdir(sys.path[0])

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

# 使用PCA进行降维
pca = PCA(n_components=3)
points_pca = pca.fit_transform(points)

# 使用t-SNE进行降维
tsne = TSNE(n_components=3)
points_tsne = tsne.fit_transform(points)

# 使用K均值聚类算法进行聚类
kmeans = KMeans(n_clusters=K)  # K是你希望的聚类数目
kmeans.fit(points_pca)
labels_kmeans = kmeans.labels_

# 使用Mean Shift聚类算法进行聚类
mean_shift = MeanShift(bandwidth=0.1)
mean_shift.fit(points_tsne)
labels_mean_shift = mean_shift.labels_

# 创建一个与原始图像相同大小的矩阵来存储分割结果
segmented_image_kmeans = np.zeros_like(image_array)
segmented_image_mean_shift = np.zeros_like(image_array)

# 为每个聚类设置不同的颜色
colors_kmeans = kmeans.cluster_centers_ / 255.0  # 将颜色范围归一化到0-1
colors_mean_shift = mean_shift.cluster_centers_ / 255.0

# 根据聚类标签对像素进行着色
for i in range(K):
    segmented_image_kmeans[labels_kmeans == i] = colors_kmeans[i] * 255  # 将颜色范围还原到0-255
    segmented_image_mean_shift[labels_mean_shift == i] = colors_mean_shift[i] * 255

# 在3D空间中显示图像分割结果和原始图像对比
fig = plt.figure(figsize=(18, 6))

# 左边展示原始图像的点集
ax1 = fig.add_subplot(131, projection='3d')
r, g, b = image_array[:, :, 0].flatten() / 255.0, image_array[:, :, 1].flatten() / 255.0, image_array[:, :, 2].flatten() / 255.0
ax1.set_title('Original Image')
ax1.scatter(r, g, b, c=image_array.reshape(-1, 3)/255.0, marker='o')
ax1.set_xlabel('Red')
ax1.set_ylabel('Green')
ax1.set_zlabel('Blue')

# 中间展示PCA降维后的K-means聚类结果
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title('K-means (PCA)')
ax2.scatter(r,g,b, c=colors_kmeans[labels_kmeans], marker='o')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')

# 右边展示t-SNE降维后的Mean Shift聚类结果
ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title('Mean Shift (t-SNE)')
ax3.scatter(r,g,b, c=colors_mean_shift[labels_mean_shift], marker='o')
ax3.set_xlabel('t-SNE 1')
ax3.set_ylabel('t-SNE 2')
ax3.set_zlabel('t-SNE 3')

plt.show()

# 可视化分割结果
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.set_title('Original Image')
ax1.imshow(image)
ax1.axis('off')

ax2.set_title('segmented image kmeans')
ax2.imshow(segmented_image_kmeans)
ax2.axis('off')

ax2.set_title('segmented image mean_shift')
ax2.imshow(segmented_image_mean_shift)
ax2.axis('off')

# now = datetime.datetime.now()
# timestamp = now.strftime('%H%M%S')

# title = 'segmented image'
# fig2.savefig('figure2/{}_{}.png'.format(title,timestamp), dpi=300)
# plt.close()

plt.show()
