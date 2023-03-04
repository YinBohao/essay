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

filename = r'data/(2023-02-21 22_49_18_2023-02-28 22_49_18).csv'

df_orig = pd.read_csv(filename, usecols=[0, 2, 3, 4], skiprows=[
                      0, 1], encoding='gbk')
df_orig['上报时间'] = pd.to_datetime(df_orig['上报时间'])
df_data_hour = df_orig.groupby(pd.Grouper(key= '上报时间', axis=0, freq='H')).mean()

for name in df_data_hour.columns:
    # print(df_data_hour[name])
    fig, axs = plt.subplots(1, 1, figsize=(15, 2))
    axs.plot(df_data_hour[name], color='blue')
    axs.set_title(name)
    plt.show()
