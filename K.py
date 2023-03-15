import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import datetime

import os
import sys
os.chdir(sys.path[0])

matplotlib.rc("font", family='Microsoft YaHei')


# def which_day(date):
#     return datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')


def re_series(grouped):
    return grouped.reset_index()['cod'].reindex(grouped.reset_index()['cod'].index[::-1]).reset_index(drop=True)


# filename = r'data/监测历史数据.xls'
filename = r'data/result1.csv'
df_boxplot = pd.read_csv(filename)
# df = pd.read_excel(filename, usecols=[0, 6], sheet_name=[0, 1, 2])
# 139267*2
# df_orig = pd.concat([df[i] for i in range(3)], ignore_index=True)

# 按周分组，去除不够一周的天数
# df_boxplot = df_boxplot.iloc[1284:].reset_index()

# 将上报时间列的字符串数据类型转化成时间戳
# df_boxplot['datetime'] = pd.to_datetime(df_boxplot['datetime'])
# 将读取数据按每小时进行分组

df_grouped_H = df_boxplot.groupby('hour')['cod'].mean()
df_grouped_H = df_grouped_H[df_grouped_H.notnull()]

# df_boxplot['hour'] = df_boxplot['datetime'].dt.hour

# 将读取数据按天进行分组
# df_grouped_D = df_orig.groupby(pd.Grouper(
#     key='datetime', axis=0, freq='D'))['cod'].mean()
# df_grouped_D = df_grouped_D[df_grouped_D.notnull()]


# df_boxplot = df_orig.copy()
# df_boxplot['date'] = df_boxplot['datetime'].dt.strftime('%Y-%m-%d')
# df_boxplot['weekday'] = df_boxplot['datetime'].apply(lambda x: x.weekday())

# 将读取数据按周进行分组
# df_grouped_W = df_boxplot.groupby(
#     pd.Grouper(key='datetime', axis=0, freq='W'))['date']


# def which_week(date):
#     for key, value in df_grouped_W:
#         if date in list(value):
#             a = str(key)[0:10]
#     return a


# df_boxplot['week'] = df_boxplot['date'].apply(lambda x: which_week(x))
# df_boxplot['which_day'] = df_boxplot['weekday'].apply(
#     lambda x: 'workday' if x < 5 else 'dayoff')

# df_boxplot.to_csv(r'data/result1.csv', index=False)
# 测试
# for i in df_boxplot['work']:
#     print(i,type(i))
# 按时刻分组,箱型图
seaborn.boxplot(y='cod', x='hour', data=df_boxplot,
                flierprops={'marker': 'o',  # 异常值形状
                            'markerfacecolor': 'red',  # 形状填充色
                            'color': 'black',  # 形状外廓颜色
                            },
                showmeans=True,
                meanline=True,)
plt.plot(df_grouped_H, color='Navy', label='mean_value')

# 按周分组,箱型图
# seaborn.boxplot(y='cod', x='week', data=df_boxplot,
#                         hue=df_boxplot['which_day'],
#                         flierprops={'marker': 'o',  # 异常值形状
#                                     'markerfacecolor': 'red',  # 形状填充色
#                                     'color': 'black',  # 形状外廓颜色
#                                     },
#                         showmeans=True,
#                         meanline=True,).invert_xaxis()

# # 按天分组,箱型图
# seaborn.boxplot(y='cod', x='date', data=df_boxplot,
#                 flierprops={'marker': 'o',  # 异常值形状
#                             'markerfacecolor': 'red',  # 形状填充色
#                             'color': 'black',  # 形状外廓颜色
#                             },
#                 showmeans=True,
#                 meanline=True,).invert_xaxis()
# plt.plot(re_series(df_grouped_D), color='g', label='mean_value')


plt.legend(fontsize=15)
# plt.xticks(rotation=90)
plt.ylabel('COD',fontsize=15)
plt.xlabel('hour',fontsize=15)
plt.show()
