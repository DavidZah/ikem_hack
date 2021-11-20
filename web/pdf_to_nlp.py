import pretty_pdf
from os import listdir
import math
import codecs
import json
import reduce

def update_statistics(data):
    for paragraph in data:
        name = paragraph[0]
        if len(paragraph[1]) == 0:
            continue
        if name in statistics:
            statistics[name] +=1
        else:
            statistics[name] = 1

def process_file(filename):
    pdf_out = pretty_pdf.mine(filename)

    dict_out = pretty_pdf.extract(pdf_out)
    print("Diagnostics: {}".format(pretty_pdf.MARKERS))
    return dict_out


def convert_to_string(input):
    text = ""

    text += " ".join(input['diagnostics']) + " "

    to_write = ['Důvod hospitalizace', 'Diagnózy', 'Průběh hospitalizace', 'Doporučená terapie', 'Doporučení', 'Z anamnézy',
     'Nynější onemocnění', 'Medikace při příjmu', 'Objektivně při příjmu', 'Z laboratorních výsledků', 'Z vyšetření']
    for key in to_write:
        line = " ".join(input[key])
        text += line + " "
    return reduce.reduce(text)

def convert_pdf_to_string(filename):
    dict_data = process_file(filename)
    return convert_to_string(dict_data)



if False:
    paths = ["../../eleven/hackathon_kd/",  "../../eleven/hackathon_kvin/", ]
    outpath = "from_pdf/"
    counter = 0
    statistics = {}
    for path_i in range(len(paths)):
        counter_i = 0
        path = paths[path_i]
        files = listdir(path)
        for file in files: #["DR009370134.pdf"]:#
            pretty_pdf.init()
            prefix = file.split(".")[0]
            if counter >= 3 and False:
                break
            counter += 1
            counter_i += 1
            filename = path+file

            dict_data = process_file(filename)

            #data = pretty_pdf.mine(filename)

            #dict_data = pretty_pdf.extract(data)
            #print("Diagnostics: {}".format(pretty_pdf.MARKERS))
            #person = pretty_pdf.person_id()
            #print(dict_data)
            person = dict_data['person']
            print("\nCompleted\t{}\t\t[{} of {}]\t path {}: {}\tPerson {}".format(
                file,
                counter_i,
                len(files),
                path_i,
                path,
                person
            ))
            print("="*20)
            if False:
                json.dump(
                    dict_data,
                    codecs.open("{}/{}.json".format(outpath,person), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False,
                        indent=4, ensure_ascii=False)

            content = convert_to_string(dict_data)
            with open("{}/{}.txt".format(outpath,person), 'w', encoding='utf-8') as f:
                f.write(content)

            #update_statistics(data)


    #get relevant pragraph names
    print("Statistics")
    for key in statistics:
        percent = math.ceil(statistics[key]/counter*10000)/100
        #print(counter)
        print("\t{}\t({}%)\t{}".format(statistics[key], percent, key))


    relevant_keys = []
    for key in statistics:
        percent = math.ceil(statistics[key] / counter * 10000) / 100
        if percent > 90:
            relevant_keys.append(key)

    print("Relevant keys:")
    print(relevant_keys)
