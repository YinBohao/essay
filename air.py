import csv

filename = r'data/(2023-02-21 22_49_18_2023-02-28 22_49_18).csv'
with open(filename) as f:
    reader = csv.reader(f)
    for i,header_nows in enumerate(reader):
        if i == 2:
            header_now = header_nows
    
    print('title')
    for index,column_header in enumerate(header_now):
        print(index,column_header)
    print('123')