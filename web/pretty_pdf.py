import pdfminer
import sys
import math
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine, LTLine
import re


NORMAL = "Tahoma"
BOLD = "Tahoma-Bold"
#from pdfminer.high_level import extract_pages
#for page_layout in extract_pages(filename):
#    for element in page_layout:
#        print(element)

PERSON_ID = None
MARKERS = []
def init():
    global PERSON_ID, MARKERS
    PERSON_ID = None
    MARKERS = []

def person_id():
    global PERSON_ID
    return PERSON_ID

def level(info):
    size = info[0]
    font = info[1]
    if size == 12 and font == BOLD:
        if align(info):
            return 1
        else:
            return 0
    elif size == 11 and font == BOLD:
        return 1
    elif size == 10 and (font == NORMAL or font == BOLD):
        return 2
    else:
        return 404

def align(info):
    return info[3]

def text(info):
    return info[2]

def check_align(start):
    aligns = [28.3465, 28.868, 43.3465, 30.8006]
    for threshold in aligns:
        if abs(start - threshold) < 0.1:
            return True

    return False

def check_marker(text, box):
    right = box[2]
    left = box[0]

    global MARKERS
    if abs(right - 552.7213) < 0.1:
        if left > 500:
            MARKERS.append(text)

def get_title(line):
    global PERSON_ID
    plain_text = line.get_text().strip()

    if not PERSON_ID:
        person = is_person_id(plain_text)
        if person:
            PERSON_ID = person

    start = line.bbox[0]
    check_marker(plain_text, line.bbox)

    is_align = check_align(start)

    size_g = None
    fontname = None
    for character in line:
        if not isinstance(character, LTChar):

            #print(character)
            continue
            print("Error 1")
            sys.exit(1)
        size = math.ceil(character.size)
        if not size_g:
            size_g = size
            fontname = character.fontname
        elif size_g != size:
            continue
            print("Error 2 = {}  not {}".format(size_g, size))
            print(plain_text)
            sys.exit(1)

    return [size_g, fontname, plain_text, is_align, line.bbox]

def mine(filename):
    reports = []

    for page_layout in extract_pages(filename):
        ignore = True
        has_title = False
        #print("\n\n==== new page ======")
        for element in page_layout:
            #print(element)
            if isinstance(element, LTTextContainer):
                group_data = []
                for text_line in element:
                    data = get_title(text_line)
                    lvl = level(data)
                    is_align = align(data)

                    if lvl == 0 and ignore:
                        has_title = True
                        #print("\n\n{}".format(text(data)))
                        #reports.append([text(data), []])

                    if ignore and is_align and lvl < 5 and not has_title:
                        if text(data) != "Pacient:" and text(data) != "Pacientka:":
                            ignore = False
                        else:
                            has_title = True

                    if lvl == 1:
                        ignore = False

                    #print(data)
                    if not ignore:
                        if lvl < 5:
                            if not is_align:
                                continue

                            if lvl == 1:
                                #print()
                                #print("Title")
                                reports.append([text(data), []])

                            #print(lvl * "\t", text(data))

                            if lvl == 2:
                                try:
                                    #print("Added")
                                    reports[-1][1].append(text(data))
                                except:
                                    print_report(reports)
                                    print("Can add {}".format(data))
                                    sys.exit(0)

                        #else:
                        #    print("too small",data)
                #print(group_data)
            #else:
            #    print(element)
    return reports


def print_report(data):
    #for title in data:
    #    print("\n\nTitle: {}".format(title[0]))
    for paragraph in data:
        print("\n{}".format(paragraph[0]))
        for line in paragraph[1]:
            print("\t\t{}".format(line))


def extract(data):
    global MARKERS, PERSON_ID
    relevant_keys = ['Důvod hospitalizace', 'Diagnózy', 'Průběh hospitalizace', 'Doporučená terapie', 'Doporučení', 'Z anamnézy',
     'Nynější onemocnění', 'Medikace při příjmu', 'Objektivně při příjmu', 'Z laboratorních výsledků', 'Z vyšetření']
    out = {}
    for key in relevant_keys:
       out[key] = []
    for paragraph in data:
        name = paragraph[0]
        if name in relevant_keys:
            out[name] = paragraph[1]

    out['diagnostics'] = MARKERS
    out['person'] = PERSON_ID
    return out

def is_person_id(str):
    m = re.search("[0-9]{6}\/[0-9]{4}", str)
    if not m:
        m = re.search("[0-9]{6}\/[0-9]{3}", str)
    if not m:
        return None
    m = m[0].replace("/", "")
    return m
