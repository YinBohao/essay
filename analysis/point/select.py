import numpy as np
import pandas as pd
import os, sys
os.chdir(sys.path[0])

filename = r'pointinfo.csv'
df = pd.read_csv(filename)

# 删除以"Arrow"开头的行
df = df[~df['Layer'].str.startswith('Arrow')]

# 根据多个字符串条件删除行
# df = df[df['Entity'] == 'Circle']

# # 创建新列并初始化为NaN
# df['point'] = np.nan
# df['geometry'] = np.nan
# # df['ponit'] = df['data_time'].apply(lambda x: this_monday(x))

# # 根据条件设置新列的值
# df.loc[(df['YS'] == df['YE']) & (df['YE'] == df['YM']), 'point'] = 1
# df.loc[(df['YS'] == df['YE']) & (df['YE'] == df['YM']), 'geometry'] = df.apply(lambda row: f"c({(row['XS'] + row['XM'])/2}, {row['YS']})", axis=1)


df.to_csv(r'pointinfo_without_Arrow.csv', encoding='gb2312', index=0)

# # 统计空值数量
# num_null_values_point = df['point'].isnull().sum()

# # 打印结果
# print("空值数量：", num_null_values_point)