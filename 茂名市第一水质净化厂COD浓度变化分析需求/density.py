import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys
os.chdir(sys.path[0])

filename = r'0711_0825_reverse.csv'
# 读取CSV文件
data = pd.read_csv(filename)

df = data

# 去除每天内的重复COD数据
df = df.drop_duplicates(subset=['date', 'COD'])

# 将日期字符串转换为日期对象
df['Date'] = pd.to_datetime(df['date'])
df['Time'] = pd.to_timedelta(df['hour'])

# 合并日期和时间以创建完整的时间戳
df['Timestamp'] = df['Date'] + df['Time']

# 按照分界日期划分数据
boundary_date = pd.Timestamp('2023-8-11')
before_boundary = df[df['Timestamp'] < boundary_date]
after_boundary = df[df['Timestamp'] >= boundary_date]

def plot_density(data1, data2, title, name):
    plt.figure(figsize=(10, 6))
    
    for hour in range(24):
        data1_count = data1[data1['Timestamp'].dt.hour == hour]['COD'].count()
        data2_count = data2[data2['Timestamp'].dt.hour == hour]['COD'].count()
        total_count1 = data1['COD'].count()
        total_count2 = data2['COD'].count()
        density1 = data1_count / total_count1
        density2 = data2_count / total_count2
        plt.bar(hour - 0.2, density1, width=0.4, alpha=0.5, color='red', label=None)
        plt.bar(hour + 0.2, density2, width=0.4, alpha=0.5, color='blue', label=None)
    
    plt.xlabel('Hour of the Day')
    plt.ylabel('Probability Density')
    plt.title(title)
    plt.xticks(np.arange(0, 24, step=1))
    
    # 设置图例标签
    plt.bar(0, 0, color='red', label='Before')
    plt.bar(0, 0, color='blue', label='After')
    plt.legend()
    
    plt.savefig(r'figure1_' + name, dpi=200)
    plt.close()

plot_density(before_boundary, after_boundary, 'COD Density Before and After 2023/8/11', 'before_and_after')
