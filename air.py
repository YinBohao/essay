import os,sys
os.chdir(sys.path[0])
import csv
from matplotlib import pyplot as plt
import numpy as np

filename = r'data/(2023-02-21 22_49_18_2023-02-28 22_49_18).csv'
import csv

def draw(filename):
    
    with open(filename,"r") as f:
        reader = csv.reader(f)
        dates,NMHCs,fumes,PMs = [],[],[],[]
        for i,header_rows in enumerate(reader):
            if i == 2:
                header_row = header_rows
            if i >= 3:
                dates.append(header_rows[0])
                NMHCs.append(float(header_rows[2]))
                fumes.append(float(header_rows[3]))
                PMs.append(float(header_rows[4]))
        f.close()

    plt.plot(dates, NMHCs, color='r', label='NMHC')
    plt.plot(dates, fumes, color='g', label='fume')
    plt.plot(dates, PMs, color='b', label='PM')

    plt.legend()
    plt.xticks([])
    # y_major_locator = MultipleLocator(0.5)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.yticks([])
    plt.show()

if __name__ == '__main__':
    draw(filename)