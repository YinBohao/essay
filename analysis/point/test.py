import os, sys
os.chdir(sys.path[0])
import pandas as pd

# 加载result.csv文件到DataFrame
result_df = pd.read_csv('result.csv')

# 设置 Pandas 显示选项，禁用科学计数法
pd.set_option('display.float_format', '{:.2f}'.format)

# 提取comparison_result为True的行的索引号
true_indices = result_df[result_df['comparison_result'] == True].index

result_df = result_df[result_df['comparison_result'] == True][['geometry', 'XS']]

# 输出数量和索引号
print("值为True的数量:", len(true_indices))
print("对应的索引号:")
print(true_indices)
print(result_df)


