import pandas as pd
import os, sys
os.chdir(sys.path[0])

import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

filename = r'data/博贺新港厂数据/csv/博贺新港厂进水数据.csv'

df = pd.read_csv(filename)

# 将上报时间列的字符串数据类型转化成时间戳
df['data_time'] = pd.to_datetime(df['data_time'])



# %W 每年的第几周，把星期一做为第一天（值从0到53）
# %U 第年的第几周，把星期日做为第一天（值从0到53）
# %V 每年的第几周，使用基于周的年
df['week'] = df['data_time'].dt.strftime('%V')

df = df.sort_values(['data_time'])

df.to_csv(r'data/博贺新港厂数据/csv/博贺新港厂进水数据1.csv',index=0)

