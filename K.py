import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

import os
import sys
os.chdir(sys.path[0])

matplotlib.rc("font", family='Microsoft YaHei')

filename = r'data/监测历史数据.xls'
# nrows8428行，3月1日-
df_orig = pd.read_excel(
    filename, usecols=[0, 6], nrows=65076, sheet_name='第一水质净化厂进水1')
# 将上报时间列的字符串数据类型转化成时间戳
df_orig['日期'] = pd.to_datetime(df_orig['日期'])
# 将读取数据按每小时进行分组
df_grouped = df_orig.groupby(pd.Grouper(key='日期', axis=0, freq='D'))['cod'].mean()
re_series=df_grouped.reset_index()['cod'].reindex(df_grouped.reset_index()['cod'].index[::-1]).reset_index(drop=True)

df_boxplot = df_orig.copy()
df_boxplot['date'] = df_boxplot['日期'].dt.strftime('%Y-%m-%d')

for name in df_boxplot.columns:
    if name not in ['日期', 'date']:
        fig, axs = plt.subplots(1, 1, figsize=(15, 10),sharex=True,sharey=True)
        seaborn.boxplot(y=name, x='date', data=df_boxplot,
                        flierprops={'marker': 'o',  # 异常值形状
                                    'markerfacecolor': 'red',  # 形状填充色
                                    'color': 'black',  # 形状外廓颜色
                                    },
                        showmeans=True,
                        meanline=True,)
        plt.xticks(rotation=45)
        # plt.show()

# group_name = [gn for gn in df_grouped.groups.keys()]
plt.plot(re_series, color='g', label='mean_value')
# plt.xticks([])
plt.legend()
plt.show()
