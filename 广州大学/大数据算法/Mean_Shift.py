import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
import numpy as np
import datetime
import os
import sys
os.chdir(sys.path[0])

# 设置设备（GPU或CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据集根目录
root = "GTSRB/train"

# 要选择的类别和每个类别的图像数量
selected_classes = ["00000", "00001", "00002", "00003", "00004"]
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

from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

# 创建Mean Shift聚类器
bandwidth = 0.1  # 带宽参数，用于控制聚类的紧密度
ms = MeanShift(bandwidth=bandwidth)

# 对RGB点集进行聚类
clusters = ms.fit_predict(img_vector)

# 绘制聚类结果
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 遍历每个点并根据聚类结果着色
for point, cluster in zip(img_vector, clusters):
    r, g, b = point[0], point[1], point[2]
    ax.scatter(r, g, b, color=plt.cm.jet(cluster / np.max(clusters)))

title = 'Mean Shift Clustering'

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title(title)

fig.tight_layout()

now = datetime.datetime.now()
timestamp = now.strftime('%H%M%S')
 
plt.savefig('figure2/{}_{}.png'.format(title,timestamp), dpi=300)
plt.close()

# plt.show()
