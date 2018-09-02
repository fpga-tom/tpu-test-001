import csv

with open('data.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(0,128):
        for j in range(0,128):
            s = i + j
            sb = [x for x in bin(int(s))[2:].zfill(8)]
            ib = [x for x in bin(int(i))[2:].zfill(8)]
            jb = [x for x in bin(int(j))[2:].zfill(8)]

            d = ib + jb + sb

            writer.writerow(d)

