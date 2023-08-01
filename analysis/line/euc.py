import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
# 假设有两个坐标点 (x1, y1) 和 (x2, y2)
x1, y1 = 93970.5,6278.356


x2, y2 = 93970.96,6278.55


# 将坐标点转换为numpy数组
coord1 = np.array([x1, y1])
coord2 = np.array([x2, y2])

# 计算欧式距离
distance = np.linalg.norm(coord1 - coord2)

# 将坐标点表示为一个矩阵，每行表示一个坐标点
coordinates = np.array([[x1, y1], [x2, y2]])

# 使用euclidean_distances函数计算欧式距离
euclidean_distances = euclidean_distances(coordinates)

# 提取欧式距离，因为只有两个坐标点，所以distances是一个2x2的矩阵
# distances[i, j] 表示第i个坐标点与第j个坐标点之间的欧式距离
euclidean_distance = euclidean_distances[0, 1]

print("欧式距离：", distance)
print("欧式距离：", euclidean_distance)
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
# import os, sys
# os.chdir(sys.path[0])
# # 假设您已经读取了CSV文件并命名为df
# name = 'HSL'

# filename_text = r'{}_text.csv'.format(name)
# filename_new = r'{}_new.csv'.format(name)

# df1 = pd.read_csv(filename_text, encoding='gb2312')
# df2 = pd.read_csv(filename_new)

# df1['x_coordinate'] = df1['x_coordinate'].astype(str).str[3:].astype(float)
# df2['XM'] = df2['XM'].astype(str).str[3:].astype(float)

# # 获取ORIG_FID为51433对应的x_coordinate、y_coordinate
# x_coord = df1.loc[df1['ORIG_FID'] == 51433, 'x_coordinate'].values[0]
# y_coord = df1.loc[df1['ORIG_FID'] == 51433, 'y_coordinate'].values[0]

# # 计算所有点之间的欧式距离矩阵
# coordinates_df1 = np.array([[x_coord, y_coord]])
# coordinates_df2 = np.array(df2[['XM', 'YM']])
# distances_matrix = euclidean_distances(coordinates_df1, coordinates_df2)

# # 获取最短距离对应的行索引
# shortest_distance_index = np.argmin(distances_matrix)

# # 获取最短距离和对应的XM、YM
# shortest_distance = np.min(distances_matrix)
# shortest_XM = df2.loc[shortest_distance_index, 'XM']
# shortest_YM = df2.loc[shortest_distance_index, 'YM']

# print("ORIG_FID为51433对应的最短距离：", shortest_distance)
# print("对应的XM和YM：", shortest_XM, shortest_YM)
# print(shortest_distance_index)



