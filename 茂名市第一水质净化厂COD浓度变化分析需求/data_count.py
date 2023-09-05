import pandas as pd
import os, sys
os.chdir(sys.path[0])

from datetime import datetime, timedelta

filename = r'0711_0825_set.csv'

df = pd.read_csv(filename)

# 将日期列转换为日期时间类型
df['date'] = pd.to_datetime(df['date'])

# 统计date列中重复数据和重复数量
# date_counts = df['date'].value_counts().reset_index()
date_counts = df.groupby('date')['COD'].count().reset_index()
# print(date_counts)

# 筛选出8月11日之后的数据
# df_after_aug_11 = df[df['date'] >= '2023-08-11']

# 去除每天内的重复COD数据
df_after_aug_11_unique = df.drop_duplicates(subset=['date', 'COD'])

df_after_aug_11_unique.to_csv(r'df_after_aug_11_unique.csv', index=0)

# 使用groupby统计每天的数据量
# daily_data_counts = df_after_aug_11_unique.groupby('date')['COD'].count().reset_index()
# 使用groupby统计每天不同hour的COD数据量
daily_hourly_data_counts = df_after_aug_11_unique.groupby(['date', 'hour'])['COD'].count().reset_index()

# 打印结果
# print(daily_hourly_data_counts)
# daily_hourly_data_counts.to_csv(r'daily_hourly_data_counts.csv', index=0)

# 打印结果
# print(daily_data_counts)

# 使用groupby统计每天中COD列的重复数据和数量
# date_cod_counts = df_after_aug_11.groupby(['date', 'COD']).size().reset_index(name='count')

# date_cod_counts.to_csv(r'count.csv', index=0)
# print(date_cod_counts)
