import pandas as pd
import os, sys
os.chdir(sys.path[0])

from datetime import datetime, timedelta

filename = r'0711_0825.csv'

df = pd.read_csv(filename)

# 将上报时间列的字符串数据类型转化成时间戳
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
df['hour'] = df['datetime'].dt.strftime('%H:00:00')

df.to_csv(r'0711_0825_set.csv', index=0)

# 测试
# for i in df_boxplot['work']:
#     print(i,type(i))