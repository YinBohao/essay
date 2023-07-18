import os, sys
os.chdir(sys.path[0])
# import pandas as pd

# # 加载pointinfo.csv文件
# df_pointinfo = pd.read_csv('pointinfo.csv')

# # 加载2.csv文件
# df_2 = pd.read_csv('2.csv')

# # 合并两个DataFrame
# merged_df = pd.merge(df_pointinfo, df_2, on='geometry', how='inner')

# # 显示合并后的DataFrame
# print(merged_df)

import pandas as pd
import numpy as np

# 加载pointinfo.csv文件到DataFrame
pointinfo_df = pd.read_csv('pointinfo_without_Arrow.csv', encoding='gb2312')

# 加载2.csv文件到DataFrame
compare_df = pd.read_csv('2.csv')

# 创建用于存储比较结果的列表
comparison_result = []

# 比较geometry列的值
for i in range(len(compare_df)):
    point_geom = compare_df.loc[i, 'geometry']  # 2.csv文件中的geometry值
    point_x, point_y = map(float, point_geom[2:-1].split(', '))  # 解析坐标值
    found_match = False
    
    for j in range(len(pointinfo_df)):
        target_geom = pointinfo_df.loc[j, 'geometry']  # pointinfo.csv文件中的geometry值
        target_x, target_y = map(float, target_geom[2:-1].split(', '))  # 解析坐标值
        
        # 计算欧氏距离（坐标误差）
        distance = np.linalg.norm([point_x - target_x, point_y - target_y])
        
        # 如果欧氏距离在0.3范围内，则认为是匹配的
        if distance <= 0.3:
            found_match = True
            break
    
    # 将比较结果添加到列表中
    comparison_result.append(found_match)

# 将比较结果添加到compare_df的新列
compare_df['comparison_result'] = comparison_result

compare_df.to_csv(r'result.csv', index=0)

# 显示结果
# print(compare_df)

