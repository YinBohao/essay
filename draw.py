import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
os.chdir(sys.path[0])

import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

filename = r'data/监测历史数据.xls'
# 读取csv文件的0，2，3，4列，并且忽略其前两行表名，从列名所在行开始
# df_orig = pd.read_csv(filename2, usecols=[0, 2, 3, 4], skiprows=[
#                       0, 1], encoding='gbk')

df_orig = pd.read_excel(
    filename, usecols=[0, 6], nrows=8428, sheet_name='第一水质净化厂进水1')

# df_orig.head(1000)

# 将上报时间列的字符串数据类型转化成时间戳
# df_orig['上报时间'] = pd.to_datetime(df_orig['上报时间'])
df_orig['日期'] = pd.to_datetime(df_orig['日期'])
# 将读取数据按每小时进行分组
# df_data_hour = df_orig.groupby(pd.Grouper(key= '上报时间', axis=0, freq='H')).mean()
df_data_hour = df_orig.groupby(pd.Grouper(key='日期', axis=0, freq='D')).mean()
# 绘图
# plt.figure(figsize=(10, 3))
plt.boxplot(df_data_hour['cod'])
# plt.plot(df_data_hour['nh3n'], color='g', label='nh3n')
# plt.plot(df_data_hour['tn'], color='b', label='tn')
# plt.plot(df_data_hour['tp'], color='y', label='tp')
# plt.plot(df_data_hour['ph'], color='c', label='ph')

# plt.xticks(rotation = 30)
plt.legend()
plt.show()
# 按天分组，箱型图
# for name in df_boxplot.columns:
#     if name not in ['日期', 'date']:
#         fig, axs = plt.subplots(1, 1, figsize=(15, 8),sharex=True,sharey=True)
#         seaborn.boxplot(y=name, x='date', data=df_boxplot,
#                         flierprops={'marker': 'o',  # 异常值形状
#                                     'markerfacecolor': 'red',  # 形状填充色
#                                     'color': 'black',  # 形状外廓颜色
#                                     },
#                         showmeans=True,
#                         meanline=True,)
#         plt.xticks(rotation=45)
#         # plt.show()

# # group_name = [gn for gn in df_grouped.groups.keys()]