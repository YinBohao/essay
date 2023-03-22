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


# # group_name = [gn for gn in df_grouped.groups.keys()]

# def which_day(date):
#     return datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')


# df = pd.read_excel(filename, usecols=[0, 6], sheet_name=[0, 1, 2])
# 139267*2
# df_orig = pd.concat([df[i] for i in range(3)], ignore_index=True)

# 将读取数据按每小时进行分组

# df_grouped_H = df_boxplot.groupby('hour')['cod'].mean()
# 去重
# df = df_boxplot.drop_duplicates(subset='datetime',keep='first',inplace=False,ignore_index=True)
# df_grouped_H.to_csv(r'data/111.csv')

# df_boxplot['day_hour'] = df_boxplot['datetime'].dt.strftime('%Y-%m-%d %H')

# df_boxplot['day_count'] = df_boxplot.groupby('date')['cod'].transform('count')

# df_boxplot = df_orig.copy()
# df_boxplot['date'] = df_boxplot['datetime'].dt.strftime('%Y-%m-%d')
# df_boxplot['weekday'] = df_boxplot['datetime'].apply(lambda x: x.weekday())

# def which_week(date):
#     for key, value in df_grouped_W:
#         if date in list(value):
#             a = str(key)[0:10]
#     return a


# df_boxplot['week'] = df_boxplot['date'].apply(lambda x: which_week(x))
# df_boxplot['which_day'] = df_boxplot['weekday'].apply(
#     lambda x: 'workday' if x < 5 else 'dayoff')

# 将读取数据按周进行分组
# df_grouped_W = df_boxplot.groupby(
#     pd.Grouper(key='datetime', axis=0, freq='W'))['date']

# 测试
# for i in df_boxplot['work']:
#     print(i,type(i))