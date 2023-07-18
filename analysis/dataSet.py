import pandas as pd
import os, sys
os.chdir(sys.path[0])

filename = r'1.csv'
df = pd.read_csv(filename)
# 根据多个字符串条件删除行
df = df[~df['Layer'].str.contains('YSP|WSP|HLP')]
df.to_csv(r'2.csv',index=0)