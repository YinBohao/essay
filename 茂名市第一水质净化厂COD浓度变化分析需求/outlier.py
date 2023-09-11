import pandas as pd
import os, sys
os.chdir(sys.path[0])

filename = r'data/0711_0825_reverse.csv'

df = pd.read_csv(filename)

# 去除每天内的重复COD数据
df = df.drop_duplicates(subset=['date', 'COD'])

# 将日期列转换为日期时间类型
df['date'] = pd.to_datetime(df['date'])

data = df

# 使用标准差方法识别异常值
mean_cod = data['COD'].mean()
std_cod = data['COD'].std()
threshold = 2  # 可根据需要调整阈值

# 识别异常值
outliers = data[(data['COD'] > mean_cod + threshold * std_cod) | (data['COD'] < mean_cod - threshold * std_cod)]

# 计算异常值数量和所占比例
num_outliers = len(outliers)
total_samples = len(data)
outlier_percentage = (num_outliers / total_samples) * 100

print(f'异常值数量：{num_outliers}')
print(f'异常值所占比例：{outlier_percentage:.2f}%')

# 删除异常值
data = data[~data.index.isin(outliers.index)]

# 保存处理后的数据
data.to_csv('cleaned_data.csv', index=False)
