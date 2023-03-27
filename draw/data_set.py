import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
os.chdir(sys.path[0])

import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

filename = r'../data/csv/水口镇水质净化厂进水数据.csv'

df = pd.read_csv(filename)

# 将上报时间列的字符串数据类型转化成时间戳
df['data_time'] = pd.to_datetime(df['data_time'])

# 将读取数据按天进行分组
df['date'] = df['data_time'].dt.strftime('%Y-%m-%d')
df['day_count'] = df.groupby('date')['cod'].transform('count')
# df['cod_mean'] = df.groupby('data_time')['cod_mean'].transform('mean')

# 将读取数据按每小时进行分组
df['day_hour'] = df['data_time'].dt.strftime('%Y-%m-%d %H')
df['hour'] = df['data_time'].dt.strftime('%H')
# # group_name = [gn for gn in df_grouped.groups.keys()]

# 将读取数据按周进行分组
df_grouped_W = df.groupby(
    pd.Grouper(key='data_time', axis=0, freq='W'))['date']
def which_week(date):
    for key, value in df_grouped_W:
        if date in list(value):
            a = str(key)[0:10]
    return a
df['weekday'] = df['data_time'].apply(lambda x: x.weekday())
df['week'] = df['date'].apply(lambda x: which_week(x))

# 删除每日数据量小于1100的日期/1440 大约75%
df = df[df['day_count']>1100]


df.to_csv(r'../data/csv/水口镇水质净化厂进水数据_result.csv',index=0)
# def which_day(date):
#     return datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')

# df_boxplot['which_day'] = df_boxplot['weekday'].apply(
#     lambda x: 'workday' if x < 5 else 'dayoff')

# 测试
# for i in df_boxplot['work']:
#     print(i,type(i))