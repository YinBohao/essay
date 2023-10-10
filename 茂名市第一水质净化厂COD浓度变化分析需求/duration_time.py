# import pandas as pd
# from datetime import datetime
# import os
# import sys
# os.chdir(sys.path[0])

# filename = r'data/0711_0825_reverse.csv'
# # 读取CSV文件
# data = pd.read_csv(filename)

# df = data
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 根据连续的COD值进行分组并计算持续时间
# df['group'] = (df['COD'] != df['COD'].shift()).cumsum()
# grouped = df.groupby('group')
# df['duration'] = grouped['datetime'].transform(lambda x: x.max() - x.min())

# # 删除用于分组的临时列
# df.drop('group', axis=1, inplace=True)

# # # 以天为单位统计每天不同COD值的持续时间段数量
# # daily_grouped = df.groupby([df['datetime'].dt.date, 'COD'])['duration'].count().reset_index()

# # # 打印结果
# # print(daily_grouped)
# # daily_grouped.to_csv('grouped_data.csv', index=False)

# # 格式化持续时间为%H:%M:%S
# # 若不格式化则为 0 days 00:40:00
# df['duration'] = df['duration'].apply(lambda x: f"{x.seconds//3600:02d}:{(x.seconds//60)%60:02d}:{x.seconds%60:02d}")

# # 保存处理后的数据到新的CSV文件
# df.to_csv('processed_data.csv', index=False)

# 每个COD持续时间段内数据量
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
os.chdir(sys.path[0])

filename = r'data/processed_data.csv'
# 读取CSV文件
data = pd.read_csv(filename)

df = data

# 将duration列转换为时间格式
df['repetition_duration'] = pd.to_timedelta(df['duration'])

# 定义时间区间
bins = [pd.Timedelta(hours=0), pd.Timedelta(hours=1), pd.Timedelta(hours=2),
        pd.Timedelta(hours=3), pd.Timedelta(hours=4), pd.Timedelta(hours=5),
        pd.Timedelta(days=365)]  # 365天作为"5小时及以上"的区间

# 将持续时间分配到不同的时间区间，并统计数量
duration_intervals = pd.cut(df['repetition_duration'], bins=bins, labels=['0-1小时', '1-2小时', '2-3小时', '3-4小时', '4-5小时', '5小时及以上'])
interval_counts = duration_intervals.value_counts()

# print(interval_counts)

# 将统计结果保存到CSV文件
# interval_counts.to_csv('data/interval_counts.csv', header=['Duration Interval', 'Count'], index_label='Duration Interval')
# 将统计结果保存到CSV文件
interval_counts_df = pd.DataFrame({'Duration Interval': interval_counts.index, 'Count': interval_counts.values})
interval_counts_df.to_csv('data/interval_counts.csv', index=False, encoding='gb2312')



