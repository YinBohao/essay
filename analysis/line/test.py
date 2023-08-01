import pandas as pd
import numpy as np
import os, sys
os.chdir(sys.path[0])

from sklearn.metrics.pairwise import euclidean_distances

# 假设您已经读取了两个CSV文件，并分别命名为df1和df2
name = 'HSL'

filename_text = r'{}_text.csv'.format(name)
filename_new = r'{}_new.csv'.format(name)

df1 = pd.read_csv(filename_text, encoding='gb2312')
df2 = pd.read_csv(filename_new)

# 将x_coordinate和y_coordinate转换为二维数组
coordinates_df1 = np.array(df1[['x_coordinate', 'y_coordinate']])
coordinates_df2 = np.array(df2[['XM', 'YM']])

# 计算所有点之间的欧式距离矩阵
distances_matrix = euclidean_distances(coordinates_df1, coordinates_df2)

# 沿着axis=0（每列）找到每个PS_HSL对应的最短距离的索引
shortest_distances_indices = np.argmin(distances_matrix, axis=0)

# 通过索引找到对应的最短距离，并保留两位有效数字
shortest_distances = distances_matrix[shortest_distances_indices, np.arange(len(df2))]
shortest_distances = np.round(shortest_distances, 5)  # 保留两位有效数字

# 将最短距离保存到df2中
df2['Shortest_Euclidean_Distance'] = shortest_distances

# 根据索引从df1中获取对应的RefName
refname_shortest = df1.loc[shortest_distances_indices, 'RefName'].values

ORIG_FID_shortest = df1.loc[shortest_distances_indices, 'ORIG_FID'].values
# 将RefName作为新列添加到df2中
df2['RefName_Shortest'] = refname_shortest

df2['ORIG_FID_Shortest'] = ORIG_FID_shortest

# 将更新后的df2保存为CSV文件
df2.to_csv(r'{}_11111111.csv'.format(name), encoding='gb2312', index=0)

df = df2

# 使用groupby和transform找到每个ORIG_FID_Shortest中最小距离对应的行
df['Shortest_Euclidean_Distance_Min'] = df.groupby('ORIG_FID_Shortest')['Shortest_Euclidean_Distance'].transform('min')

# 将不满足条件的ORIG_FID_Shortest和Shortest_Euclidean_Distance置空
df.loc[df['Shortest_Euclidean_Distance'] != df['Shortest_Euclidean_Distance_Min'], ['ORIG_FID_Shortest', 'Shortest_Euclidean_Distance']] = [None, None]

# 删除用于辅助计算的最小距离列
df.drop(columns=['Shortest_Euclidean_Distance_Min'], inplace=True)

# 假设您已经读取了更新后的CSV文件，并命名为df_updated
df_updated = df

# # 划分距离列为不同的区间，并添加一个新的列'Distance_Range'表示区间
# distance_bins = [0, 0.3, 0.5, 0.8, 1, float('inf')]  # 区间端点，左闭右开
# distance_labels = ['0-0.3', '0.3-0.5', '0.5-0.8', '0.8-1', '1+']  # 对应的区间标签
# df_updated['Distance_Range'] = pd.cut(df_updated['Shortest_Euclidean_Distance'], bins=distance_bins, labels=distance_labels, right=False)

# # 统计每个区间内的数量
# distance_counts = df_updated['Distance_Range'].value_counts()

# print("距离列各个范围内的数量：")
# print(distance_counts)

# 统计Shortest_Euclidean_Distance列的值及其频数，并获取前5位数据
top_5_distances = df_updated['Shortest_Euclidean_Distance'].value_counts().head(10)

print("Shortest_Euclidean_Distance列重复数量前5位的数据：")
print(top_5_distances)
print(top_5_distances.sum())
