import pandas as pd
import numpy as np
import os, sys
os.chdir(sys.path[0])

from sklearn.metrics.pairwise import euclidean_distances

# 找出每一列的'XM', 'YM'对应的距离最短的text
name = 'YSL'

filename_text = r'{}_text.csv'.format(name)
filename_new = r'{}_new.csv'.format(name)

df1 = pd.read_csv(filename_text, encoding='gb2312')
df2 = pd.read_csv(filename_new)

df1['x_coordinate'] = df1['x_coordinate'].astype(str).str[3:].astype(float)

df2['XM'] = df2['XM'].astype(str).str[3:].astype(float)

# 将x_coordinate和y_coordinate转换为二维数组
coordinates_df1 = np.array(df1[['x_coordinate', 'y_coordinate']])
coordinates_df2 = np.array(df2[['XM', 'YM']])

# 计算所有点之间的欧式距离矩阵
distances_matrix = np.round(euclidean_distances(coordinates_df1, coordinates_df2), decimals=1)

# 沿着axis=0（每列）找到每个PS_HSL对应的最短距离的索引
shortest_distances_indices = [np.where(distances_matrix[:, i] == 0.5)[0] for i in range(distances_matrix.shape[1])]

df2['Shortest_Euclidean_Distance'] = [0.5 if len(idx_list) > 0 else np.nan for idx_list in shortest_distances_indices]
# # 为 df2 添加新列 Shortest_Euclidean_Distance，初始值设置为空
# df2['Shortest_Euclidean_Distance'] = np.nan

# 获取对应的 RefName 列，并将其作为新列添加到 df2 中
ref_names = [df1.loc[idx_list, 'RefName'].tolist() if len(idx_list) > 0 else [np.nan] for idx_list in shortest_distances_indices]
df2['RefName'] = [names[0] for names in ref_names]

orig_fids = [df1.loc[idx_list, 'ORIG_FID'].tolist() if len(idx_list) > 0 else [np.nan] for idx_list in shortest_distances_indices]
df2['ORIG_FID'] = [fids[0] for fids in orig_fids]

# 将更新后的df2保存为CSV文件
df2.to_csv(r'{}_oneone.csv'.format(name), encoding='gb2312', index=0)

# 在 df2 中去除 ORIG_FID 列中值为空的行
df2 = df2.dropna(subset=['ORIG_FID'])
df2.to_csv(r'{}_oneone_without_nan.csv'.format(name), encoding='gb2312', index=0)

#############################################################
df2['ORIG_FID'] = df2['ORIG_FID'].astype(int)

# 将ORIG_FID_Shortest列中的所有非空值转换为set
existing_values_set = set(df2['ORIG_FID'].dropna())

# 生成包含所有应有值的range
expected_values_range = set(range(df2['ORIG_FID'].min(), df2['ORIG_FID'].max() + 1))

# 使用差集操作找到缺失的值
missing_values = expected_values_range - existing_values_set

print("缺少的ORIG_FID值：", missing_values)

##########################################
# 使用 isnull() 方法判断 ID 列中的空值，并使用 np.where() 方法获取空值的索引
null_id_indices = np.where(df2['ID'].isnull())[0]
print("ID列空值索引：",null_id_indices)
# df2['ID'] = df2['ID'].astype(int)

# # 将ORIG_FID_Shortest列中的所有非空值转换为set
# existing_values_set = set(df2['ID'].dropna())

# # 生成包含所有应有值的range
# expected_values_range = set(range(df2['ID'].min(), df2['ID'].max() + 1))

# # 使用差集操作找到缺失的值
# missing_values = expected_values_range - existing_values_set

# print("缺少的ID值：", missing_values)

# 统计Shortest_Euclidean_Distance列的值及其频数，并获取前5位数据
top_5_distances = df2['Shortest_Euclidean_Distance'].value_counts().head(10)

print("Shortest_Euclidean_Distance列重复数量前5位的数据：")
print(top_5_distances)
# print(top_5_distances.sum())

# 统计 ORIG_FID 列中每个值的出现次数
orig_fid_counts = df2['ORIG_FID'].value_counts()

# 找出出现次数大于等于 2 的值
orig_fid_values_greater_than_or_equal_to_2 = orig_fid_counts[orig_fid_counts >= 2].index.tolist()
print("重复的ORIG_FID：",orig_fid_values_greater_than_or_equal_to_2)

id_counts = df2['ID'].value_counts()
id_values_greater_than_or_equal_to_2 = id_counts[id_counts >= 2].index.tolist()
print("重复的ID：",id_values_greater_than_or_equal_to_2)








# # 使用groupby和idxmin找到每个ORIG_FID_Shortest中最小距离对应的索引
# # min_distance_indices = df.groupby('ORIG_FID_Shortest')['Shortest_Euclidean_Distance'].idxmin()
# # 根据索引筛选出最短距离对应的行，并保存到新的DataFrame
# # df2 = df.loc[min_distance_indices]

# # 按'ORIG_FID_Shortest'进行分组，并保留组内'Shortest_Euclidean_Distance'为0.5的行
# grouped_df = df.groupby('ORIG_FID_Shortest').filter(lambda x: (x['Shortest_Euclidean_Distance'] == 0.5).any())

# # 获取满足条件的行的索引
# indices = grouped_df.groupby('ORIG_FID_Shortest').apply(lambda x: x[x['Shortest_Euclidean_Distance'] == 0.5].index)

# # 展开索引列表
# flat_indices = [idx for sublist in indices.tolist() for idx in sublist]

# # 使用loc方法获取对应的行
# df2 = df.loc[flat_indices]

# # 将筛选后的DataFrame保存为CSV文件
# # df2.to_csv(r'{}_oneone.csv'.format(name), encoding='gb2312', index=0)
