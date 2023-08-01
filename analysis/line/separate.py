import pandas as pd
import os, sys
os.chdir(sys.path[0])

name = 'WSL'

# filename = r'{}_oneone.csv'.format(name)
filename = r'{}_oneone_without_nan.csv'.format(name)
df = pd.read_csv(filename, encoding='gb2312')

df[['Texture', 'DN']] = df['RefName'].str.split('  ', expand=True)

df['Pipe_Diameter'] = df['DN'].str[2:]
df['DN'] = df['DN'].str[:2]

# 打印DataFrame查看结果
# df.to_csv(r'{}_oneone.csv'.format(name), encoding='gb2312', index=0)
df.to_csv(r'{}_oneone_without_nan.csv'.format(name), encoding='gb2312', index=0)
