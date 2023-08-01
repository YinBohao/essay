import pandas as pd
import os, sys
os.chdir(sys.path[0])

name = 'HSL'

filename = r'annotationinfo.csv'
df = pd.read_csv(filename)
df = df[df['Layer'].str.contains(name)]
df.to_csv(r'{}_text.csv'.format(name), encoding='gb2312', index=0)