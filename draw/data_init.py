import pandas as pd
import matplotlib

import os
import sys
os.chdir(sys.path[0])

matplotlib.rc("font", family='Microsoft YaHei')
matplotlib.use('TkAgg')

# filename = r'../data/博贺新港厂数据/博贺镇水质净化厂首期工程进水数据.xls'
# wb = pd.read_excel(
#     filename, usecols=[0, 7],sheet_name=[0,1,2,3,4,5,6],header=None)
# wb = pd.concat([wb[i] for i in range(len(wb))], ignore_index=True)

# wb.to_csv(r'../data/博贺新港厂数据/csv/博贺镇水质净化厂首期工程进水数据.csv',header=0,index=0)

# filename1 = r'../data/csv/水口镇水质净化厂进水数据.csv'
# filename2 = r'../data/csv/黄岭镇水质净化厂进水数据.csv'
filename = r'../data/博贺新港厂数据/csv/博贺镇水质净化厂首期工程进水数据.csv'
df = pd.read_csv(filename)
# print(df[df['cod'] == 0])

# 去空值
# df = df.drop(df[df['cod'] <= 0].index)
df = df.dropna(how="all")
# df.to_csv(r'../data/博贺新港厂数据/csv/博贺新港厂进水数据1.csv',index=0)

# df['data_time'] = pd.to_datetime(df['data_time'])

# 去重
# df = df.drop_duplicates(subset='data_time',keep='first',inplace=False,ignore_index=True)
# df = df.sort_values(['data_time'])

# df.to_csv(r'../data/博贺新港厂数据/csv/博贺镇水质净化厂首期工程进水数据1.csv',index=0)
# 去重 ABAB型
# df = df.drop_duplicates(subset='data_time',keep='first',inplace=False,ignore_index=True)
# df.to_csv(r'../data/csv/黄岭镇水质净化厂进水数据_first.csv',index=0)

# df = df.drop_duplicates(subset='data_time',keep='last',inplace=False,ignore_index=True)
# df.to_csv(r'../data/csv/黄岭镇水质净化厂进水数据_last.csv',index=0)
