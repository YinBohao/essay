import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

import os
import sys
os.chdir(sys.path[0])

matplotlib.rc("font", family='Microsoft YaHei')
matplotlib.use('TkAgg')

filename = r'data/0711_0825_reverse.csv'
df_boxplot = pd.read_csv(filename)

# 去除每天内的重复COD数据
df_boxplot = df_boxplot.drop_duplicates(subset=['date', 'COD'])

# 绘制图表类型
# name = 'date'
name = 'hour'
# name = 'week'
# name = ['date', 'hour', 'week']

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(30, 20))

# 绘制箱型图和均值线
def plot_grouped_data(group_col):
    # group_col = ['date', 'hour', 'week']

    seaborn.boxplot(y='COD', x=group_col, data=df_boxplot,
                    color='white',
                    flierprops={'marker': 'o',  # 异常值形状
                                'markerfacecolor': 'red',  # 形状填充色
                                'color': 'black',  # 形状外廓颜色
                                },
                    medianprops={'color': 'green'},
                    showmeans=True,
                    meanprops={'marker': 'o',
                               'markerfacecolor': 'b',
                               'markeredgecolor': 'b'},
                    )
    
    # 均值线
    grouped_data = df_boxplot.groupby(group_col)['COD'].mean()
    plt.plot(grouped_data, color='blue', label='mean_value')

plot_grouped_data(name)

if name == 'date':
    # 在8月11日处绘制分界线
    aug_11 = '2023-08-11'
    plt.axvline(x=aug_11, color='orange', linestyle='--')
    
    # 设置x轴分界线刻度标签
    xticks = ax.get_xticks()
    xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]

    for i, label in enumerate(xtick_labels):
        if label == aug_11:
            ax.get_xticklabels()[i].set_color("orange")

plt.axhline(y = 150, color = 'r', linestyle = '--')
# 设置y轴警戒线刻度标签
extraticks=[150]
plt.yticks(list(plt.yticks()[0]) + extraticks)

yticks = ax.get_yticks()
ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]

for i, label in enumerate(ytick_labels):
    if label == '150':
        ax.get_yticklabels()[i].set_color("red")

title_size = 30
font_size = 15

plt.title('进水数据', fontsize = title_size)
plt.legend(fontsize=title_size)
plt.xticks(rotation=90, size=font_size)

plt.yticks(size=font_size)
plt.ylabel('COD', fontsize=title_size)
plt.xlabel(name, fontsize=title_size)

plt.savefig(r'figure/figure1_' + name + '.png', pad_inches = 0.5, dpi =200)

plt.close()

