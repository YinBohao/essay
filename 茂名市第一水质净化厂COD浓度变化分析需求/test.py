import pandas as pd
import os, sys
os.chdir(sys.path[0])

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pylab
from scipy.stats import shapiro, anderson

filename = r'0711_0825_reverse.csv'

df = pd.read_csv(filename)

# 去除每天内的重复COD数据
df = df.drop_duplicates(subset=['date', 'COD'])

# 将日期列转换为日期时间类型
df['date'] = pd.to_datetime(df['date'])

# 创建条件以分割数据
date_threshold = pd.to_datetime('2023-08-11')
df_before_aug_11 = df[df['date'] < date_threshold]  # 8月11日之前的数据
df_after_aug_11 = df[df['date'] >= date_threshold]  # 8月11日之后的数据

# 8月11日之前的数据 - 数据分布可视化
data_to_test = df_before_aug_11['COD']

plt.figure(figsize=(8, 6))
sns.histplot(data_to_test, kde=True)
plt.title('Histogram of COD Data (Before Aug 11)')
plt.xlabel('COD Values')
plt.ylabel('Frequency')
plt.savefig(r'Before_Histogram', pad_inches = 0.5, dpi =200)
# plt.show()

# plt.figure(figsize=(8, 6))
# sm.qqplot(data_to_test, line='s', markersize=5)
# plt.title('Q-Q Plot (Before Aug 11)')
# pylab.show()

# 正态性检验 - Shapiro-Wilk 测试
_, p_value = shapiro(data_to_test)
alpha = 0.05  # 设置显著性水平
if p_value > alpha:
    print("8月11日之前的数据符合正态分布")
else:
    print("8月11日之前的数据不符合正态分布")

# 8月11日之后的数据 - 数据分布可视化
data_to_test = df_after_aug_11['COD']

plt.figure(figsize=(8, 6))
sns.histplot(data_to_test, kde=True)
plt.title('Histogram of COD Data (After Aug 11)')
plt.xlabel('COD Values')
plt.ylabel('Frequency')
plt.savefig(r'After_Histogram', pad_inches = 0.5, dpi =200)
# plt.show()
plt.close()

# plt.figure(figsize=(8, 6))
# sm.qqplot(data_to_test, line='s', markersize=5)
# plt.title('Q-Q Plot (After Aug 11)')
# pylab.show()

# 正态性检验 - Shapiro-Wilk 测试
_, p_value = shapiro(data_to_test)
alpha = 0.05  # 设置显著性水平
if p_value > alpha:
    print("8月11日之后的数据符合正态分布")
else:
    print("8月11日之后的数据不符合正态分布")
