import os,sys
os.chdir(sys.path[0])
import csv
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

filename = r'data/(2023-02-21 22_49_18_2023-02-28 22_49_18).csv'

with open(filename) as f:
    reader = csv.reader(f)
    for i,header_rows in enumerate(reader):
        if i == 2:
            header_row = header_rows
    print(header_row)
    print('title')
    # for index,column_header in enumerate(header_now):
    #     print(index,column_header)
    dates,NMHCs,fumes,PMs = [],[],[],[]
    for row in reader:
        dates.append(row[0])
        NMHCs.append(row[2])
        fumes.append(row[3])
        PMs.append(row[4])
    print(dates)
    f.close()
