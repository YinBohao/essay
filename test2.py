import pandas as pd
import os
import sys
os.chdir(sys.path[0])

filename = r'data/监测历史数据.xls'
df = pd.ExcelFile(filename)


df_concat= pd.concat([ pd.read_excel(df, sheet) for sheet in df.sheet_names])
#将所有sheet中数据合并到一个df中