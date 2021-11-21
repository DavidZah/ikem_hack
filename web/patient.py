from ast import parse
import numpy as np
import json
import re
import csv
import os
from utils import *
from io import StringIO
from pathlib import Path
from csv_to_npy import *
from xml.dom import minidom
import pickle
"""
Patient:
This class currently has three possible constructors that fit the three different ways in which we were building the training dataset during the hackathon
The first takes the specific json_file with identificators and classifications and a npy_file that has fitting ECG data
The second takes only the data - it is used for simple testing
The final one takes a csv_file of the large data set and parses it to a readable 2D list, for each parsed identificator it looks for an xml file and an nlp file to upload data
"""
class Patient:
    def __init__(self):
        self.data = None
        self.identificator = None
        self.classification = None
        self.nlp = None
        self.type = 3

    def generate_from_json(self, npy_file, json_file) -> None:
        raw_data = self.import_data(npy_file, json_file)
        self.identificator = self.data[0]
        self.classification = self.data[1]
        self.data = raw_data[2]
        self.nlp = None
        self.type = 2
    
    def generate_from_ecg(self, data) -> None:
        self.data = data
        self.type = 2
    
    def generate_from_pdf(self, text) -> None:
        self.nlp = text
        self.type = 1 
    
    def generate_from_pdf_and_ecg(self, text, data) -> None:
        self.nlp = text
        self.data = data
        self.type = 0
    
    def generate_from_xmls(self, csv_data, nlp_folder, xml_folder, save_xml=False):
        self.classification = csv_data[1]
        self.identificator = csv_data[0]
        self.data = self.find_xml(self.identificator, xml_folder, ".xml")
        if self.data[0][0] != None and save_xml:
            self.save_npy(npy_folder, self.identificator, self.data)
        self.nlp = self.find_file(self.identificator, nlp_folder, ".txt")
        self.type = self.get_type()

    def generate_from_npy(self, csv_data, nlp_folder, npy_folder) -> None:
        self.classification = csv_data[1]
        self.identificator = csv_data[0]
        self.data = self.find_npy(self.identificator, npy_folder, ".npy")
        self.nlp = self.find_file(self.identificator, nlp_folder, ".txt")
        self.type = self.get_type()

    def get_type(self):
        if self.nlp is None and self.data is None:
            return 3
        elif self.data is None:
            return 1
        elif self.nlp is None:
            return 2
        else:
            return 0

    def find_npy(self, identificator, folder, extension):
        dir_list = os.listdir(str(folder))
        for id in range(len(dir_list)):
            if identificator + extension == dir_list[id]:
                return np.load(str(folder.joinpath(identificator + extension)))
        return None

    def find_xml(self, identificator, folder, extension):
        dir_list = os.listdir(str(folder))
        for id in range(len(dir_list)):
            if identificator + extension == dir_list[id]:
                stream = StringIO()
                print(identificator + extension)
                ret = parse_xml(str(folder.joinpath(identificator + extension)), stream)
                if ret == None:
                    return None
                return generate_numpy(stream.getvalue())
        return None
    
    def find_file(self, identificator, folder, extension):
        dir_list = os.listdir(str(folder))
        for id in range(len(dir_list)):
            if identificator + extension == dir_list[id]:
                f = open(str(folder.joinpath(identificator + extension)), 'rb')
                ret = f.read()
                f.close()
                return ret
        return None

    def import_data(self, npy_file, json_file):
        json_data = []
        f = json.load(open(json_file,'r'))
        for row in f:
            json_data.append(row)
        with open(npy_file, 'rb') as f:
            ecg = np.load(f)
        identificator = npy_file.split('/')[-1]
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

    def save_npy(self, npy_folder_path, patient, data):
        with open(str(npy_folder_path.joinpath(patient+".npy")), 'wb') as f:
            inp = data
            inp = np.transpose(inp)
            np.save(f, inp)


if __name__ == "__main__":
    csv_file = Path("data/dgs.csv")
    nlp_folder =  Path("data/from_pdf")
    xml_folder = Path("data/MUSE_20211007_143634_97000")
    npy_folder = Path("data/npy")

    data = parse_csv_file(csv_file.absolute())
    patients = []
    for i in data:
        current_patient = Patient(i, nlp_folder, npy_folder)
        if(current_patient.type != 3):
            patients.append(current_patient)

    with open('data/parrot.pkl', 'wb') as f:
        pickle.dump(patients, f)
