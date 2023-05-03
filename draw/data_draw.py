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
# 均值线x轴反转
def re_series(grouped):
    return grouped.reset_index()['cod'].reindex(grouped.reset_index()['cod'].index[::-1]).reset_index(drop=True)


# 处理过后的数据 _result.csv
# filename0 = r'../data/csv/黄岭镇水质净化厂进水数据_first_result.csv'
# filename1 = r'../data/csv/黄岭镇水质净化厂进水数据_last_result.csv'
# filename2 = r'../data/csv/黄岭镇水质净化厂进水数据_mean_result.csv'
# filename3 = r'../data/csv/水口镇水质净化厂进水数据_result.csv'
filename = r'../data/博贺新港厂数据/csv/博贺新港厂进水数据_init_drop.csv'
df_boxplot = pd.read_csv(filename)

df_boxplot = df_boxplot.sort_values(['data_time'], ascending=False)
# df_boxplot = df_boxplot.sort_values(['hour'], ascending=True)

# df_boxplot['week'] = pd.to_datetime(df_boxplot['week']).dt.strftime('%Y-%m-%d')

# 按周分组，去除不够一周的天数
# 删除每周数据量小于2160*5 = 10800的日期 不足5天
df_boxplot = df_boxplot[df_boxplot['week_count']>10800]

# 将上报时间列的字符串数据类型转化成时间戳
df_boxplot['data_time'] = pd.to_datetime(df_boxplot['data_time'])

plt.figure(
    # figsize = (30,5), # 设置图片大小
    figsize = (30,20), # 设置图片大小
    # dpi = 80        # 精度
)

# 按天分组均值线
df_grouped_D = df_boxplot.groupby('date')['cod'].mean()

# 按时刻分组均值线
df_grouped_H = df_boxplot.groupby('hour')['cod'].mean()

# 按周分组均值线
df_grouped_W = df_boxplot.groupby('week')['cod'].mean()

# 按时刻分组,箱型图
# seaborn.boxplot(y='cod', x='hour', data=df_boxplot,
#                 color='white',
#                 flierprops={'marker': 'o',  # 异常值形状
#                             'markerfacecolor': 'red',  # 形状填充色
#                             'color': 'black',  # 形状外廓颜色
#                             },
#                 medianprops={'color': 'green'},
#                 showmeans=True,
#                 meanprops={'marker': 'o',
#                             'markerfacecolor': 'b',
#                             'markeredgecolor': 'b'},)
# plt.plot(df_grouped_H, color='blue', label='mean_value')

# 按周分组,箱型图
seaborn.boxplot(y='cod', x='week', data=df_boxplot,
                color='white',
                # hue=df_boxplot['which_day'],
                flierprops={'marker': 'o',  # 异常值形状
                            'markerfacecolor': 'red',  # 形状填充色
                            'color': 'black',  # 形状外廓颜色
                            },
                medianprops={'color': 'green'},
                showmeans=True,
                meanprops={'marker': 'o',
                           'markerfacecolor': 'b',
                           'markeredgecolor': 'b'},).invert_xaxis()
plt.plot(re_series(df_grouped_W), color='blue', label='mean_value')

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
#                 # ax=axs
#                 ).invert_xaxis()

# plt.plot(re_series(df_grouped_D), color='blue', label='mean_value')

plt.axhline(y = 150, color = 'r', linestyle = '--')

# # plt.text(-0.9, 150, '150', fontsize = 10, color = 'red')

title_size = 30
font_size = 15
name = 'week'
plt.title('博贺新港厂进水数据', fontsize = title_size)
plt.legend(fontsize=title_size)
plt.xticks(rotation=90, size=font_size)
# fontproperties = 'Times New Roman', 
# plt.xticks(range(0, len(re_series(df_grouped_D)), 2))
plt.yticks(size=font_size)
plt.ylabel('COD', fontsize=title_size)
plt.xlabel(name, fontsize=title_size)

plt.savefig(r'D:\实习\essay\figure\博贺新港厂进水数据\figure1_' + name,pad_inches = 0.5,dpi =200)
# ,bbox_inches = 'tight'
plt.close()
# plt.show()
