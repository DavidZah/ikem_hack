import os

import numpy as np
import json

class Patient:
    def __init__(self, npy_file, json_file) -> None:
        self.data = self.import_data(npy_file, json_file)
        self.identificator = self.data[0]
        self.classification = self.data[1]
        self.ecg = self.data[2]
        self.nlp = None

    def import_data(self, npy_file, json_file):
        json_data = []
        f = json.load(open(json_file,'r'))
        for row in f:
            json_data.append(row)
        with open(npy_file, 'rb') as f:
            ecg = np.load(f)
        identificator = npy_file.split(os.path.sep)[-1]
        identificator = identificator.split('.')[0]
        classification = self.find_classification(identificator, json_data)
        if classification == -1:
            return [None, None, None]
        return [identificator, classification, ecg]
    
    def find_classification(self, matcher, arr):
        for id in range(len(arr)):
            if int(matcher) == int(arr[id][0]):
                return arr[id][1]
        return -1

        
            
