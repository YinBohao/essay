import pandas as pd
import os, sys
os.chdir(sys.path[0])
# 假设您已经读取了CSV文件并命名为df
name = 'HSL'

filename =  r'{}_new.csv'.format(name)

df = pd.read_csv(filename, encoding='gb2312')

import pandas as pd

# 假设已经读取了数据并存储在名为df的DataFrame中

# 提取x_coordinate列中每个元素的前三位
x_coordinate_prefix = df['YM'].astype(str).str[:3]

# 找出不同的值
different_values = x_coordinate_prefix.unique()

# 找出不同值对应的索引
different_values_indices = {value: x_coordinate_prefix[x_coordinate_prefix == value].index.tolist() for value in different_values}

# 输出结果
print("不同的值：", different_values)
print(x_coordinate_prefix.value_counts())
# print("不同值对应的索引：", different_values_indices)



