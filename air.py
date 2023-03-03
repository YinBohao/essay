import os,sys
os.chdir(sys.path[0])
import csv
from matplotlib import pyplot as plt

filename = r'data/(2023-02-21 22_49_18_2023-02-28 22_49_18).csv'
import csv

with open(filename,"r") as f:
    reader = csv.reader(f)
    dates,NMHCs,fumes,PMs = [],[],[],[]
    for i,header_rows in enumerate(reader):
        if i == 2:
            header_row = header_rows
        if i in range(3,100):
            dates.append(header_rows[0])
            NMHCs.append(header_rows[2])
            fumes.append(header_rows[3])
            PMs.append(header_rows[4])
    f.close()

plt.plot(dates, NMHCs, color='r', label='NMHC')
plt.plot(dates, fumes, color='g', label='fume')
plt.plot(dates, PMs, color='b', label='PM')

plt.legend()
plt.xticks([])
plt.yticks([])
plt.show()