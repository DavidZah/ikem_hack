import os
from patient import Patient

def get_training_set(json_file, npy_path):
    return_list = []
    npy_filelist = os.listdir(npy_path)
    count = 0
    for filename in npy_filelist:
        new_patient = Patient(npy_path+filename, json_file)
        if new_patient.classification == None:
            continue
        count += 1
        return_list.append(new_patient)
    print(count)
    return return_list