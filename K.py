import os,sys
os.chdir(sys.path[0])

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

filename = r'data/监测历史数据.xls'
# 8428行，3月1日-
df_orig = pd.read_excel(
    filename, usecols=[0, 6], nrows=4108, sheet_name='第一水质净化厂进水1')
# 将上报时间列的字符串数据类型转化成时间戳
# df_orig['上报时间'] = pd.to_datetime(df_orig['上报时间'])
df_orig['日期'] = pd.to_datetime(df_orig['日期'])
# 将读取数据按每小时进行分组
# df_data_hour = df_orig.groupby(pd.Grouper(key= '上报时间', axis=0, freq='H')).mean()
df_data_hour = df_orig.groupby(pd.Grouper(key='日期', axis=0, freq='D'))
M,N,ss = [],[],[]
for key, value in df_data_hour:
    M.append(str(key))
    # print(key)
    value_li = value['cod'].values.tolist()
    for i in value_li:
        ss.append(i)
    N.append(ss)
print(M)
print(N)
# plt.boxplot(N,labels=M)
# plt.xticks(rotation = 30)
# plt.legend()
# plt.show()