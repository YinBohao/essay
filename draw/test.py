import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import datetime
import matplotlib.ticker as ticker

import os
import sys
os.chdir(sys.path[0])

matplotlib.rc("font", family='Microsoft YaHei')
matplotlib.use('TkAgg')

filename = r'../data/博贺新港厂数据/csv/博贺新港厂进水数据_init.csv'

df = pd.read_csv(filename)

df['data_time'] = pd.to_datetime(df['data_time'])

# 计算数据开始出现的时间和相同数据出现的持续时间
df['data_start_time'] = df['data_time'].where(df['cod'] != df['cod'].shift(1))

# 对 data_start_time 列的 NaN 值进行向前填充
df['data_start_time'].ffill(inplace=True)

df['data_duration'] = df['data_time'] - df['data_start_time']

# 将 data_duration 列转换为字符串格式（HH:MM:SS）
df['data_duration_hour'] = df['data_duration'].dt.total_seconds().astype(int).apply(lambda x: round(x / 3600, 4))
df['data_duration'] = df['data_duration'].dt.total_seconds().astype(int).apply(lambda x: pd.to_datetime(x, unit='s').strftime('%H:%M:%S'))

# 删除不连续的数据行
# df = df.dropna(subset=['data_start_time'])

# 输出结果
# print(df)

# filename = r'../data/博贺新港厂数据/csv/博贺新港厂进水数据_init111.csv'

# df = pd.read_csv(filename)

df = df.drop_duplicates(subset='data_start_time',keep='last',inplace=False,ignore_index=True)

df.to_csv(r'../data/博贺新港厂数据/csv/博贺新港厂进水数据_init_drop.csv',index=0)

# import pandas as pd

# # 创建一个示例DataFrame
# df = pd.DataFrame({'data': [1, 1, 1, 2, 3, 3, 3, 3],
#                    'time': ['2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04', '2023-04-05', '2023-04-06', '2023-04-07', '2023-04-08']})

# # 将时间列转换为日期时间类型
# df['time'] = pd.to_datetime(df['time'])

# # 计算相同数据出现的持续时间，并转换为HH:MM:SS格式
# df['data_duration'] = df.groupby((df['data'] != df['data'].shift(1)).cumsum())['time'].apply(lambda x: (x.max() - x.min()).total_seconds())
# df['data_duration'] = pd.to_datetime(df['data_duration'], unit='s').dt.strftime('%H:%M:%S')

# # 计算数据开始出现的时间
# df['data_start_time'] = df.groupby((df['data'] != df['data'].shift(1)).cumsum())['time'].min()

# # 重置索引
# df.reset_index(drop=True, inplace=True)

# print(df)
