import wave
from os import listdir
import numpy as np
import csv
import re

path = "csvs2/"
outpath = "npy2/"

def find_diagnosis(strin):
    if strin[0:3] == "I35":
        return 1
    else:
        return 0

def parse_csv_file(filename):
    data = []
    with open(str(filename), newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        first = True
        for row in spamreader:
            temp_line = []
            if first:
                first = False
                continue
            row_data = []
            for item in row:
                val = item.strip()
                if val == "":
                    continue
                row_data.append(val)
                row_data = row_data[0:2]
            identificator = re.sub("[a-zA-Z]", row_data[0], "")
            temp_line.append(row_data[0].replace("/",""))
            temp_line.append(row_data[1].replace("\"", ""))
            temp_line[1] = find_diagnosis(temp_line[1])
            data.append(temp_line)
    return data

def save_npy(patient, name, data):
    with open(outpath+str(patient)+".npy", 'wb') as f:
        inp = np.array(data)
        inp = np.transpose(inp)
        print(inp.shape)
        np.save(f, inp)


"""
counter = 0
for file in listdir(path):
    prefix = file.split(".")[0]
    print(file)
    counter += 1
    if counter > 10 and False:
        break
    filename = path+file
    data = from_csv(filename)
    n = np.array(data)
    print(n)
    save_npy(prefix, "", n)
    #save_npy(counter, "rhythm", rhythm)

"""

