import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
os.chdir(sys.path[0])

matplotlib.rc("font", family='Microsoft YaHei')

filename = r'data/监测历史数据.xls'
# 8428行，3月1日-
df_orig = pd.read_excel(
    filename, usecols=[0, 6], nrows=30000, sheet_name='第一水质净化厂进水1')
# 将上报时间列的字符串数据类型转化成时间戳
# df_orig['上报时间'] = pd.to_datetime(df_orig['上报时间'])
df_orig['日期'] = pd.to_datetime(df_orig['日期'])
# 将读取数据按每小时进行分组
# df_data_hour = df_orig.groupby(pd.Grouper(key= '上报时间', axis=0, freq='H')).mean()
df_data_hour = df_orig.groupby(pd.Grouper(key='日期', axis=0, freq='D'))
M, N = [], []
for key, value in df_data_hour:
    M.append(str(key)[0:10])
    # print(key)
    value_li = value['cod'].values.tolist()
    N.append(value_li)
# print(M)
# print(N)
plt.figure(figsize=(15, 4))
plt.boxplot(N, labels=M,
            flierprops={'marker': 'o',  # 异常值形状
                        'markerfacecolor': 'red',  # 形状填充色
                        'color': 'black',  # 形状外廓颜色
                        },
            )
plt.xticks(rotation=20)
# plt.xticks([])
# plt.legend()
plt.show()
