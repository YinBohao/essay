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

filename = r'../data/博贺新港厂数据/csv/博贺镇水质净化厂首期工程进水数据_init.csv'

df = pd.read_csv(filename)

# df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

df = df[df['date'] == '2022-10-09']
df = df[(df['hour'] == '00:00:00') | (df['hour'] == '01:00:00') | (df['hour'] == '02:00:00') | (df['hour'] == '03:00:00')]

# df.to_csv(r'../data/博贺新港厂数据/csv/博贺镇水质净化厂首期工程进水数据_init11111.csv',index=0)


plt.figure(
    # figsize = (30,5), # 设置图片大小
    figsize = (40,8), # 设置图片大小
    # dpi = 80        # 精度
)

title_size = 30
font_size = 15
name = 'duration2'
plt.plot(df['data_time'],df['cod'])

# 根据 y 值大小显示或隐藏 x 轴刻度  
# if df['data_duration_hour'].astype(float) < 1.8:  
#     plt.xticks(df['data_start_time'].min(), df['data_start_time'].max(), rotation=90)  
# elif df['data_duration_hour'].astype(float) > 2:  
#     plt.xticks(df['data_start_time'].min(), df['data_start_time'].max(), rotation=0)  
# else:  
#     plt.xticks([])
 

plt.title('博贺镇水质净化厂首期工程进水数据', fontsize = title_size)
# plt.xticks([])
# fontproperties = 'Times New Roman', 
plt.xticks(range(0, len(df['data_time']), 8))
plt.xticks(rotation=90, size=font_size)
plt.yticks(size=font_size)
plt.savefig(r'D:\实习\essay\figure\博贺镇水质净化厂首期工程进水数据\figure_' + name,pad_inches = 0.5,dpi =200)
# ,bbox_inches = 'tight'
plt.close()
# plt.show()