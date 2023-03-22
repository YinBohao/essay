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

# 均值线x轴反转
def re_series(grouped):
    return grouped.reset_index()['cod'].reindex(grouped.reset_index()['cod'].index[::-1]).reset_index(drop=True)

# filename = r'data/监测历史数据.xls'

# 处理过后的数据 result.csv
filename = r'data/result.csv'
df_boxplot = pd.read_csv(filename)

# 按周分组，去除不够一周的天数
# indexNames=df_boxplot[(df_boxplot['week'] == '2023/3/12')].index
# df_boxplot.drop(indexNames,inplace=True)
# indexNames=df_boxplot[(df_boxplot['week'] == '2023/1/8')].index
# df_boxplot.drop(indexNames,inplace=True)

# 将上报时间列的字符串数据类型转化成时间戳
df_boxplot['datetime'] = pd.to_datetime(df_boxplot['datetime'])


# 按天分组均值线
df_grouped_D = df_boxplot.groupby(pd.Grouper(
    key='datetime', axis=0, freq='D'))['cod'].mean()
df_grouped_D = df_grouped_D[df_grouped_D.notnull()]

# 按时刻分组均值线
# df_grouped_H = df_boxplot.groupby('hour')['cod'].mean()

# 删除每日数据量小于1100的日期/1440 大约75%
# df_boxplot[df_boxplot['day_count']>1100].to_csv(r'data/result1.csv', index=False)

# 按时刻分组,箱型图
# seaborn.boxplot(y='cod', x='hour', data=df_boxplot,
#                 flierprops={'marker': 'o',  # 异常值形状
#                             'markerfacecolor': 'red',  # 形状填充色
#                             'color': 'black',  # 形状外廓颜色
#                             },
#                 showmeans=True,
#                 meanline=True,)
# plt.plot(df_grouped_H, color='Navy', label='mean_value')

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
seaborn.boxplot(y='cod', x='date', data=df_boxplot,
                flierprops={'marker': 'o',  # 异常值形状
                            'markerfacecolor': 'red',  # 形状填充色
                            'color': 'black',  # 形状外廓颜色
                            },
                showmeans=True,
                meanline=True,).invert_xaxis()
plt.plot(re_series(df_grouped_D), color='g', label='mean_value')


plt.legend(fontsize=15)
plt.xticks(rotation=90)
plt.ylabel('COD',fontsize=15)
plt.xlabel('hour',fontsize=15)
plt.show()
