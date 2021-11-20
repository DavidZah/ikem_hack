from io import StringIO
import numpy as np
import json, codecs
import os
from pathlib import Path
from io import StringIO
import wave
import re
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def find_identificator_from_pdf(filepath):
    identificator = ""
    if os.path.isfile(str(filepath)):
        text = convert_pdf_to_string(filepath)
        identificator = find_xml_filename(text)
        if identificator is None:
            return None
    else:
        return None
    return [identificator, text]

def convert_pdf_to_string(file_path):
	output_string = StringIO()
	with open(str(file_path), 'rb') as in_file:
	    parser = PDFParser(in_file)
	    doc = PDFDocument(parser)
	    rsrcmgr = PDFResourceManager()
	    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
	    interpreter = PDFPageInterpreter(rsrcmgr, device)
	    for page in PDFPage.create_pages(doc):
	        interpreter.process_page(page)
	return(output_string.getvalue())

def find_xml_filename(pdftext):
    m = re.compile("[0-9]{6}\/[0-9]{4}?")
    m = list(filter(m.match, pdftext))
    if len(m) == 0:
        m = re.compile("[0-9]{6}\/[0-9]{3}?")
        m = list(filter(m.match, pdftext))
    if len(m) == 0:
        return None
    m = m[0].replace("/", "")
    return m

def load_from_json(filepath):
    arr = []
    data = json.load(open(filepath,'r'))
    for i in data:
        arr.append(i)
    return arr

def find_diagnosis(pdftext, diagnosis):
    if any(diagnosis in s for s in pdftext):
        return 1
    else:
        return 0

def check_if_id_is_known(identificator, arr):
    for i in range(len(arr)):
        if arr[i][0] == identificator:
            return i
    return 0

def process_pdf(pdf_file_path, json_file_path):
    res = find_identificator_from_pdf(pdf_file_path)
    if res is None:
        return None
    identificator = res[0]
    text = res[1]
    if not os.path.isfile(str(json_file_path)):
        return None
    arr = load_from_json(json_file_path)
    id = check_if_id_is_known(identificator, arr)
    if id == 0:
        diagnosis = find_diagnosis(text, "I35")
        arr.append([identificator, diagnosis])
        json.dump(arr, codecs.open(json_file_path, 'w', encoding='utf-8'), separators=(',',':'), sort_keys=True, indent=4)
    else:
        diagnosis = arr[id][1]

    return [identificator, diagnosis]

def save_npy(npy_folder_path, patient, data):
    with open(str(npy_folder_path.joinpath(patient+".npy")), 'wb') as f:
        inp = np.array(data)
        inp = np.transpose(inp)
        np.save(f, inp)

def generate_numpy(input):
    data = input.split("\n")
    data.pop(0)
    data.pop()
    for i in range(len(data)):
        data[i] = data[i].split(",")
        data[i].pop()
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j].strip())
    np_data = np.array(data)
    return np_data

def parse_xml(file_path, output_string):
    prefix = file_path.split(os.sep)[-1].split(".")[0]
    if os.path.isfile(str(file_path)):
        try:
            wave.process(file_path, output_string)
        except:
            return None
        return prefix
    else:
        return None