import scipy.io as scio
import pandas as pd
import csv


data=scio.loadmat('./MOS/mosz2.mat')
mos = data["MOSz"]

csv_file_path = "mos_authenticity.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['name', 'mos'])

    # Write the data rows
    for i in range(len(mos)):
        name = str(i) + '.png'
        row = (name, mos[i].item())
        writer.writerow(row)
