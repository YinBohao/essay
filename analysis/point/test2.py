import os, sys
os.chdir(sys.path[0])
import pandas as pd

import pandas as pd

# 加载pointinfo.csv文件到DataFrame
pointinfo_df = pd.read_csv('3.csv', encoding='gb2312')

# 加载2.csv文件到DataFrame
compare_df = pd.read_csv('222.csv')

# 提取pointinfo.csv中X列和Y列的整数部分
pointinfo_df['X_int'] = pointinfo_df['X'].astype(int)
pointinfo_df['Y_int'] = pointinfo_df['Y'].astype(int)

# 提取2.csv中X列和Y列的整数部分
compare_df['X_int'] = compare_df['X'].astype(int)
compare_df['Y_int'] = compare_df['Y'].astype(int)

# 创建新列并初始化为NaN
compare_df['pointinfo_index'] = float('nan')

# 在2.csv中查找与pointinfo.csv中元素相等的行索引号
for i in range(len(compare_df)):
    X_int = compare_df.loc[i, 'X_int']
    Y_int = compare_df.loc[i, 'Y_int']
    matching_indices = pointinfo_df[(pointinfo_df['X_int'] == X_int) & (pointinfo_df['Y_int'] == Y_int)].index
    if len(matching_indices) > 0:
        compare_df.loc[i, 'pointinfo_index'] = matching_indices[0]

# 显示结果
compare_df.to_csv(r'result222.csv', index=0)

# 提取pointinfo_index非空行的索引号
non_empty_indices = compare_df[compare_df['pointinfo_index'].notnull()].index

# 显示结果
print("pointinfo_index非空行的索引号:")
print(non_empty_indices)
# print(compare_df)
