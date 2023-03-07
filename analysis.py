import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
import os
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

os.chdir(sys.path[0])


warnings.filterwarnings("ignore")

filename1 = r'data/(2023-02-20 00_00_00_2023-02-26 00_00_00).csv'
filename2 = r'data/(2023-02-27 00_00_00_2023-03-05 00_00_00).csv'
filename3 = r'data/监测历史数据.xls'
# 读取csv文件的0，2，3，4列，并且忽略其前两行表名，从列名所在行开始
# df_orig = pd.read_csv(filename2, usecols=[0, 2, 3, 4], skiprows=[
#                       0, 1], encoding='gbk')

df_orig = pd.read_excel(
    filename3, usecols=[0, 1, 2, 3, 5, 6], sheet_name='第一水质净化厂进水1')
df_orig.head(1000)

# 将上报时间列的字符串数据类型转化成时间戳
# df_orig['上报时间'] = pd.to_datetime(df_orig['上报时间'])
df_orig['日期'] = pd.to_datetime(df_orig['日期'])
# 将读取数据按每小时进行分组
# df_data_hour = df_orig.groupby(pd.Grouper(key= '上报时间', axis=0, freq='H')).mean()
df_data_hour = df_orig.groupby(pd.Grouper(key='日期', axis=0, freq='H')).mean()
# 绘图
plt.figure(figsize=(10, 3))
# plt.plot(df_data_hour['非甲烷总烃(mg/m3)'],color='r', label='NMHC')
# plt.plot(df_data_hour['油烟浓度(mg/m3)'],color='g', label='fume')
# plt.plot(df_data_hour['颗粒物(mg/m3)'],color='b', label='PM')
plt.plot(df_data_hour['cod'], color='r', label='cod')
plt.plot(df_data_hour['nh3n'], color='g', label='nh3n')
plt.plot(df_data_hour['tn'], color='b', label='tn')
plt.plot(df_data_hour['tp'], color='y', label='tp')
plt.plot(df_data_hour['ph'], color='c', label='ph')

# plt.xticks(rotation = 30)
plt.legend()
# 保存文件
# plt.savefig('(2023-02-27 00_00_00_2023-03-05 00_00_00)')
plt.show()
