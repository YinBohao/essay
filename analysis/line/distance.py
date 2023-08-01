import pandas as pd
import numpy as np
import os, sys
os.chdir(sys.path[0])

from sklearn.metrics.pairwise import euclidean_distances

# 假设您已经读取了两个CSV文件，并分别命名为df1和df2
name = 'HSL'

filename_text = r'{}_text_en.csv'.format(name)
filename_new = r'{}_new.csv'.format(name)

df1 = pd.read_csv(filename_text)
df2 = pd.read_csv(filename_new)

# 将x_coordinate和y_coordinate转换为二维数组
coordinates_df1 = np.array(df1[['x_coordinate', 'y_coordinate']])
coordinates_df2 = np.array(df2[['XM', 'YM']])

# 计算所有点之间的欧式距离矩阵
distances_matrix = euclidean_distances(coordinates_df1, coordinates_df2)

# 沿着axis=0（每列）找到每个PS_HSL对应的最短距离的索引
shortest_distances_indices = np.argmin(distances_matrix, axis=0)

# 通过索引找到对应的最短距离
shortest_distances = distances_matrix[shortest_distances_indices, np.arange(len(df2))]
shortest_distances = np.round(shortest_distances, 2)  # 保留两位有效数字

# 将最短距离保存到df2中
df2['Shortest_Euclidean_Distance'] = shortest_distances

# 将更新后的df2保存为CSV文件
df2.to_csv(r'{}.csv'.format(name), index=0)


