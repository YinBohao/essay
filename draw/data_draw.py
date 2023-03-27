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
matplotlib.use('TkAgg')
# 均值线x轴反转


def re_series(grouped):
    return grouped.reset_index()['cod'].reindex(grouped.reset_index()['cod'].index[::-1]).reset_index(drop=True)


# 处理过后的数据 _result.csv
filename0 = r'../data/csv/黄岭镇水质净化厂进水数据_first_result.csv'
filename1 = r'../data/csv/黄岭镇水质净化厂进水数据_last_result.csv'
filename2 = r'../data/csv/黄岭镇水质净化厂进水数据_mean_result.csv'
filename3 = r'../data/csv/水口镇水质净化厂进水数据_result.csv'
df_boxplot = pd.read_csv(filename3)

# df_boxplot['week'] = pd.to_datetime(df_boxplot['week']).dt.strftime('%Y-%m-%d')

# 按周分组，去除不够一周的天数
# indexNames=df_boxplot[(df_boxplot['week'] == '2023-03-19')].index
# df_boxplot.drop(indexNames,inplace=True)
# indexNames=df_boxplot[(df_boxplot['week'] == '2022-12-25')].index
# df_boxplot.drop(indexNames,inplace=True)

# 将上报时间列的字符串数据类型转化成时间戳
df_boxplot['data_time'] = pd.to_datetime(df_boxplot['data_time'])

# 按天分组均值线
# df_grouped_D0 = df_boxplot0.groupby(pd.Grouper(
#     key='data_time', axis=0, freq='D'))['cod'].mean()
# df_grouped_D0 = df_grouped_D0[df_grouped_D0.notnull()]

# df_grouped_D1 = df_boxplot1.groupby(pd.Grouper(
#     key='data_time', axis=0, freq='D'))['cod'].mean()
# df_grouped_D1 = df_grouped_D1[df_grouped_D1.notnull()]

# df_grouped_D = df_boxplot.groupby(pd.Grouper(
#     key='data_time', axis=0, freq='D'))['cod'].mean()
# df_grouped_D = df_grouped_D[df_grouped_D.notnull()]

# 按时刻分组均值线
df_grouped_H = df_boxplot.groupby('hour')['cod'].mean()

# 按周分组均值线
# df_grouped_W = df_boxplot.groupby(pd.Grouper(
#     key='data_time', axis=0, freq='1W'))['cod'].mean()

# fig, axs = plt.subplots(nrows=2, ncols=1,sharex=True)

# 按时刻分组,箱型图
seaborn.boxplot(y='cod', x='hour', data=df_boxplot,
                color='white',
                flierprops={'marker': 'o',  # 异常值形状
                            'markerfacecolor': 'red',  # 形状填充色
                            'color': 'black',  # 形状外廓颜色
                            },
                medianprops={'color': 'green'},
                showmeans=True,
                meanprops={'marker': 'o',
                            'markerfacecolor': 'b',
                            'markeredgecolor': 'b'},)
plt.plot(df_grouped_H, color='blue', label='mean_value')

# 按周分组,箱型图
# seaborn.boxplot(y='cod', x='week', data=df_boxplot,
#                 color='white',
#                 # hue=df_boxplot['which_day'],
#                 flierprops={'marker': 'o',  # 异常值形状
#                             'markerfacecolor': 'red',  # 形状填充色
#                             'color': 'black',  # 形状外廓颜色
#                             },
#                 medianprops={'color': 'green'},
#                 showmeans=True,
#                 meanprops={'marker': 'o',
#                            'markerfacecolor': 'b',
#                            'markeredgecolor': 'b'},).invert_xaxis()
# plt.plot(re_series(df_grouped_W), color='blue', label='mean_value')

# 按天分组,箱型图
# seaborn.boxplot(y='cod', x='date', data=df_boxplot,
#                 color='white',
#                 flierprops={'marker': 'o',  # 异常值形状
#                             'markerfacecolor': 'red',  # 形状填充色
#                             'color': 'black',  # 形状外廓颜色
#                             },
#                 medianprops={'color': 'green'},
#                 showmeans=True,
#                 meanprops={'marker': 'o',
#                            'markerfacecolor': 'b',
#                            'markeredgecolor': 'b'},
#                 # ax=axs[0]
#                 ).invert_xaxis()
# # axs[0].plot(re_series(df_grouped_D0), color='blue', label='mean_value')
# plt.plot(re_series(df_grouped_D), color='blue', label='mean_value')

# # axs[0].axhline(y = 150, color = 'r', linestyle = '--')
# plt.axhline(y = 150, color = 'r', linestyle = '--')

# seaborn.boxplot(y='cod', x='date', data=df_boxplot1,
#                 color='white',
#                 flierprops={'marker': 'o',  # 异常值形状
#                             'markerfacecolor': 'red',  # 形状填充色
#                             'color': 'black',  # 形状外廓颜色
#                             },
#                 medianprops={'color': 'green'},
#                 showmeans=True,
#                 meanprops={'marker': 'o',
#                            'markerfacecolor': 'b',
#                            'markeredgecolor': 'b'},
#                 # ax=axs[1]
#                 ).invert_xaxis()
# # axs[1].plot(re_series(df_grouped_D1), color='blue', label='mean_value')
# plt.plot(re_series(df_grouped_D1), color='blue', label='mean_value')

# # axs[1].axhline(y = 150, color = 'r', linestyle = '--')
# plt.axhline(y = 150, color = 'r', linestyle = '--')
# # plt.text(-0.9, 150, '150', fontsize = 10, color = 'red')

# seaborn.boxplot(y='cod_mean', x='date', data=df_boxplot2,
#                 color='white',
#                 flierprops={'marker': 'o',  # 异常值形状
#                             'markerfacecolor': 'red',  # 形状填充色
#                             'color': 'black',  # 形状外廓颜色
#                             },
#                 medianprops={'color': 'green'},
#                 showmeans=True,
#                 meanprops={'marker': 'o',
#                            'markerfacecolor': 'b',
#                            'markeredgecolor': 'b'},
#                 # ax=axs[1]
#                 ).invert_xaxis()
# # axs[1].plot(re_series(df_grouped_D1), color='blue', label='mean_value')
# plt.plot(re_series(df_grouped_D2), color='blue', label='mean_value')

# # axs[1].axhline(y = 150, color = 'r', linestyle = '--')
# plt.axhline(y = 150, color = 'r', linestyle = '--')
# # plt.text(-0.9, 150, '150', fontsize = 10, color = 'red')

# for i in range(2):
#     axs[i].legend(fontsize=15)
#     axs[i].set_xlabel('day',fontsize=15)
#     axs[i].set_ylabel('COD',fontsize=15)

# axs[0].set_title('黄岭镇水质净化厂进水数据_first')
# axs[1].set_title('黄岭镇水质净化厂进水数据_last')
plt.title('水口镇水质净化厂进水数据', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(rotation=90)
plt.ylabel('COD', fontsize=15)
plt.xlabel('hour', fontsize=15)
plt.show()
