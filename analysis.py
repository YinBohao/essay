import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

import os
import sys
os.chdir(sys.path[0])

import warnings
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

filename1 = r'data/(2023-02-21 22_49_18_2023-02-28 22_49_18).csv'
filename2 = r'data/(2023-02-25 18_51_15_2023-03-04 18_51_15).csv'


df_orig = pd.read_csv(filename2, usecols=[0, 2, 3, 4], skiprows=[
                      0, 1], encoding='gbk')
df_orig['上报时间'] = pd.to_datetime(df_orig['上报时间'])
df_data_hour = df_orig.groupby(pd.Grouper(key= '上报时间', axis=0, freq='H')).mean()

plt.figure(figsize=(10,3))
plt.plot(df_data_hour['非甲烷总烃(mg/m3)'],color='r', label='NMHC')
plt.plot(df_data_hour['油烟浓度(mg/m3)'],color='g', label='fume')
plt.plot(df_data_hour['颗粒物(mg/m3)'],color='b', label='PM')
# plt.xticks(rotation = 30)
plt.legend()
plt.show()

# for name in df_data_hour.columns:
#     fig, axs = plt.subplots(1, 1, figsize=(10, 2))
#     axs.plot(df_data_hour[name], color='blue')
#     axs.set_title(name)
#     plt.show()
