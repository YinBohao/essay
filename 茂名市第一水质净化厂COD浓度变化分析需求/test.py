import pandas as pd
import os, sys
os.chdir(sys.path[0])

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# filename = r'data/0711_0825_reverse.csv'
filename = r'data/cleaned_data.csv'

df = pd.read_csv(filename)

# 去除每天内的重复COD数据
# df = df.drop_duplicates(subset=['date', 'COD'])

# 将日期列转换为日期时间类型
df['date'] = pd.to_datetime(df['date'])

# 创建条件以分割数据
date_threshold = pd.to_datetime('2023-08-11')
df_before_aug_11 = df[df['date'] < date_threshold]  # 8月11日之前的数据
df_after_aug_11 = df[df['date'] >= date_threshold]  # 8月11日之后的数据

# 选择要比较的两组数据
data_before = df_before_aug_11['COD']
data_after = df_after_aug_11['COD']

# 可视化比较
plt.figure(figsize=(10, 6))
sns.histplot(data_before, kde=True, label='Before Aug 11')
sns.histplot(data_after, kde=True, label='After Aug 11')
plt.title('Histogram of COD Data')
plt.xlabel('COD Values')
plt.ylabel('Frequency')
plt.legend()

# plt.savefig(r'Histogram_cleaned', pad_inches = 0.5, dpi =200)
# plt.close()
plt.show()

# 统计测试 - Mann-Whitney U 测试
_, p_value_mw = mannwhitneyu(data_before, data_after)
alpha = 0.05  # 设置显著性水平

print("Mann-Whitney U 测试结果：")
print(f"Mann-Whitney U P 值: {p_value_mw:.8f}")  # 输出 P 值并保留6位小数
if p_value_mw < alpha:
    print("8月11日前后的数据在中位数上存在显著差异")
else:
    print("8月11日前后的数据在中位数上没有显著差异")
