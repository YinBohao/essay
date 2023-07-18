import pandas as pd
import os, sys
os.chdir(sys.path[0])

filename = r'lineinfo.csv'
df = pd.read_csv(filename, usecols=[1,2,8,9,10,11,12,13])
# 创建一个布尔索引，表示每一行是否包含中文
contains_chinese = df['Layer'].str.contains('[\u4e00-\u9fff]', regex=True)
# 根据布尔索引删除包含中文的行
df = df[~contains_chinese]
# df.head(100)
# print(df)
# df.info()
df.to_csv(r'1.csv',index=0)

# a = (((37493456.1478 + 37493455.2196) / 2 - 37493455.472) ** 2 + ((2394676.85682 + 2394676.8344) / 2 - 2394676.8405) ** 2) ** 0.5 
# print(a)