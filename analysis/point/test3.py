import pandas as pd
import numpy as np
import os, sys
os.chdir(sys.path[0])

# 加载pointinfo.csv文件到DataFrame
pointinfo_df = pd.read_csv('3.csv', encoding='gb2312')

# 加载2.csv文件到DataFrame
compare_df = pd.read_csv('222.csv')

# 设置阈值
threshold = 0.3

# 创建新列并初始化为NaN
compare_df['pointinfo_index'] = float('nan')

# 遍历2.csv中的坐标点
for i in range(len(compare_df)):
    x = compare_df.loc[i, 'X']
    y = compare_df.loc[i, 'Y']
    matching_indices = []
    
    # 遍历pointinfo.csv中的坐标点
    for j in range(len(pointinfo_df)):
        point_x = pointinfo_df.loc[j, 'X']
        point_y = pointinfo_df.loc[j, 'Y']
        
        # 计算欧氏距离
        distance = np.sqrt((x - point_x) ** 2 + (y - point_y) ** 2)
        
        # 判断是否在阈值范围内
        if distance < threshold:
            matching_indices.append(j)
    
    # 将匹配的行索引号存储在pointinfo_index列中的对应行
    if len(matching_indices) > 0:
        compare_df.loc[i, 'pointinfo_index'] = ','.join(map(str, matching_indices))

# 显示结果
compare_df.to_csv(r'result111.csv', index=0)

# 提取pointinfo_index非空行的索引号
non_empty_indices = compare_df[compare_df['pointinfo_index'].notnull()].index

# 显示结果
print("pointinfo_index非空行的索引号:")
print(non_empty_indices)
