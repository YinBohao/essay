import pandas as pd
import numpy as np
import os, sys
os.chdir(sys.path[0])

PS = 'PS_YSL'
Arrow = 'ArrowPS_YSL'


filename = r'init_without_point.csv'
df = pd.read_csv(filename)
df = df[df['Layer'].isin([PS, Arrow])]

# 定义给定误差
tolerance = 0.25

# 创建新列并初始化为NaN
df['ID'] = np.nan

# 获取所有PS_YSL行的索引
ps_ysl_indices = df[df['Layer'] == PS].index

# 迭代PS_YSL行的索引
counter = 1
for idx in ps_ysl_indices:
    row = df.loc[idx]
    matching_rows = df[(df['Layer'] == Arrow) &
                           (((((df['XS'] + df['XM']) / 2 - row['XM']) ** 2 + ((df['YS'] + df['YM']) / 2 - row['YM']) ** 2) ** 0.5) < tolerance)]
        
  
    
    # 检查是否找到匹配的行
    if len(matching_rows) > 0:
        # 给予相同编号
        df.loc[idx, 'ID'] = counter
        df.loc[matching_rows.index, 'ID'] = counter
        counter += 1

# 打印结果
# print(df)
df.to_csv(r'4.csv',index=0)
# 统计空值数量
num_null_values = df['ID'].isnull().sum()

# 打印结果
print("欧氏距离误差: ", tolerance)
print("编号列中的空值数量：", num_null_values)
print(PS, Arrow,"diff: ", len(df[df['Layer'] == PS]), len(df[df['Layer'] == Arrow]), len(df[df['Layer'] == PS]) - len(df[df['Layer'] == Arrow]))

# 找出编号列重复最多的数字和重复次数
most_common_number = df['ID'].mode().values[0]
repeated_count = df['ID'].value_counts()[most_common_number]

# 打印结果
print(f"编号列重复最多的数字是: {most_common_number}")
print(f"重复次数: {repeated_count}")

# 统计某列重复值的数量
value_counts = df['ID'].value_counts()

# 统计每个重复次数对应的值的数量
count_per_duplicates = value_counts.value_counts()

# 打印出重复数量小于m的具体值

values_less_than_m1 = value_counts[value_counts == 1].index.tolist()

m = 2
values_less_than_m2 = value_counts[value_counts > m].index.tolist()

print("每个重复次数的值的数量:")
print(count_per_duplicates)

print("重复数量小于", m, "的具体值:")
print(values_less_than_m1)

print("重复数量大于", m, "的具体值:")
print(values_less_than_m2)

# # 统计某列重复值的数量
# value_counts = df['ID'].value_counts()

# # 统计重复了3次的值的数量
# count_3_duplicates = len(value_counts[value_counts == 3])

# # 统计重复了2次的值的数量
# count_2_duplicates = len(value_counts[value_counts == 2])

# # 统计重复了1次的值的数量
# count_1_duplicates = len(value_counts[value_counts == 1])

# print("重复了3次的值的数量:", count_3_duplicates)
# print("重复了2次的值的数量:", count_2_duplicates)
# print("重复了1次的值的数量:", count_1_duplicates)





