import time
import csv
import matplotlib.pyplot as plt
import os
import sys
os.chdir(sys.path[0])

current_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
print(current_date)
filename = r'data/(2023-02-21 22_49_18_2023-02-28 22_49_18).csv'
with open(filename) as f:
    reader = csv.reader(f)
    for i, header_rows in enumerate(reader):
        if i == 2:
            header_row = header_rows
    print(header_row)

    report_time = []
    NMHCs = []
    fumes = []
    PMs = []

    for row in reader:
        report_time.append(row[1])
        NMHCs.append(row[3])
        fumes.append(row[4])
        PMs.append(row[5])
# 根据数据绘制图形
plt.title("air", fontsize=24)
y = [NMHCs, fumes, PMs]
cnames = ['NMHC', 'fume', 'PM']
for a in range(3):
    colors = ["blue", "red", "yellow"]
    print(y[a])
    plt.plot(report_time, y[a], color=colors[a], label=cnames[a])
plt.xlabel('report_time')
plt.show()
