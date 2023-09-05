import pandas as pd

import os
import sys
os.chdir(sys.path[0])

filename = r'0711_0825.xlsx'
wb = pd.read_excel(filename, usecols=[0, 6], header=0)

wb.to_csv(r'0711_0825.csv', header=0, index=0, encoding='gb2312')

df = pd.read_csv(r'0711_0825.csv', encoding='gb2312')

# 修改列名
new_column_names = ['COD', 'datetime']  # 新的列名列表
df.columns = new_column_names


df.to_csv(r'0711_0825.csv', index=0)

# import pandas as pd
# import os, sys
# os.chdir(sys.path[0])

# import pandas as pd

# # 读取CSV文件
# file_path = '0711_0825_set.csv'  # 替换为你的文件路径
# df = pd.read_csv(file_path)

# # 倒序数据并重置索引
# df_reversed = df.iloc[::-1].reset_index(drop=True)

# # 将倒序后的数据保存到新的CSV文件
# reversed_file_path = '0711_0825_reverse.csv'  # 替换为你想保存的文件路径
# df_reversed.to_csv(reversed_file_path, index=False)





