import pandas as pd
import os, sys
os.chdir(sys.path[0])

name = 'HSL'

filename = r'{}_text.csv'.format(name)
df = pd.read_csv(filename, encoding='gb2312', usecols=[1,4,9,10])

df['RefName'] = df['RefName'].str.replace('砼', 'tong')  # 将"中文"替换为实际中文字符和"English"替换为相应的英文字符串
df['RefName'] = df['RefName'].str.replace('铸铁', 'zhutie')  # 将"中文"替换为实际中文字符和"English"替换为相应的英文字符串


# # 提取x坐标
# df['x_coordinate'] = df['geometry'].str.extract(r'c\((.*?),')

# # 提取y坐标
# df['y_coordinate'] = df['geometry'].str.extract(r', (.*?)\)')

# # 将坐标转换为浮点数
# df['x_coordinate'] = df['x_coordinate'].astype(float) 
# df['y_coordinate'] = df['y_coordinate'].astype(float)

# 设置 Pandas 显示选项，禁用科学计数法
# pd.set_option('display.float_format', '{:.3f}'.format)


# df.to_csv(r'H_text.csv', encoding='gb2312', index=0)
df.to_csv(r'{}_text_en.csv'.format(name), index=0)