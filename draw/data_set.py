import pandas as pd
import os, sys
os.chdir(sys.path[0])

from datetime import datetime, timedelta
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

filename = r'../data/博贺新港厂数据/csv/博贺新港厂进水数据.csv'

df = pd.read_csv(filename)

# 将上报时间列的字符串数据类型转化成时间戳
df['data_time'] = pd.to_datetime(df['data_time'])
df['date'] = df['data_time'].dt.strftime('%Y-%m-%d')
df['hour'] = df['data_time'].dt.strftime('%H:00:00')
df['weekday'] = df['data_time'].dt.strftime('%A')

def this_monday(today):
    """
    :function: 获取本周周一日期
    :param today:
    :return: 返回周一的日期
    :return_type: string
    """
    return datetime.strftime(today - timedelta(today.weekday()), '%Y-%m-%d')

# %W 每年的第几周，把星期一做为第一天（值从0到53）
# %U 第年的第几周，把星期日做为第一天（值从0到53）
# %V 每年的第几周，使用基于周的年
df['week'] = df['data_time'].apply(lambda x: this_monday(x))

df['day_count'] = df.groupby(pd.Grouper(key='data_time', axis=0, freq='D'))['cod'].transform('count')

df['week_count'] = df.groupby('week')['cod'].transform('count')

# 删除每日数据量小于2160的日期/2880 75%
df = df[df['day_count']>2160]
df = df.sort_values(['data_time'])

df.to_csv(r'../data/博贺新港厂数据/csv/博贺新港厂进水数据_init.csv',index=0)

# 将读取数据按天进行分组
# df['date'] = df['data_time'].dt.strftime('%Y-%m-%d')
# df['day_count'] = df.groupby('date')['cod'].transform('count')
# df['cod_mean'] = df.groupby('data_time')['cod_mean'].transform('mean')

# 将读取数据按每小时进行分组
# df['day_hour'] = df['data_time'].dt.strftime('%Y-%m-%d %H')
# # group_name = [gn for gn in df_grouped.groups.keys()]

# 将读取数据按周进行分组
# def which_week(date):
#     for key, value in df_grouped_W:
#         if date in list(value):
#             a = str(key)[0:10]
#     return a
# df['weekday'] = df['data_time'].apply(lambda x: x.weekday())
# df['week'] = df['date'].apply(lambda x: which_week(x))


# def which_day(date):
#     return datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')

# df_boxplot['which_day'] = df_boxplot['weekday'].apply(
#     lambda x: 'workday' if x < 5 else 'dayoff')

# 测试
# for i in df_boxplot['work']:
#     print(i,type(i))