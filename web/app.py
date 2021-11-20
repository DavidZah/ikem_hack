from datetime import date
from flask import send_from_directory

today = date.today()
from flask import render_template

from flask import Flask, flash, request
from werkzeug.utils import secure_filename
import random
from io import StringIO
from dataset_use import Predictor
from utils import *
from patient import Patient
import pdf_to_nlp
import os


UPLOAD_FOLDER = Path("tmp")
pdf_file=""
xml_file=""
json_data = None


predictor = Predictor(str(Path("classifier/final.h5")),str(Path("classifier/nlp_model.h5")),str(Path("../data/nlp_vectorizer.pkl")))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == extension

def do_ai_magic():
    return random.randrange(0,100)

def process_file(request, filename, extension):
    if len(request.files) == 0:
        return 1
    elif filename not in request.files:
        return 1
    file = request.files[filename]
    if not allowed_file(file.filename, extension):
        flash("Selected file is invalid")
        return 2
    else:
        pdf_file = secure_filename(file.filename)
        file.save(str(app.config['UPLOAD_FOLDER'].joinpath(pdf_file)))
        return 0

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if "UploadPDF" in request.form:
            # check if the post request has the file part
            resPDF = process_file(request, "filePDF", "pdf")
            if resPDF == 0:
                pdf_file = secure_filename(request.files["filePDF"].filename)
                content = pdf_to_nlp.convert_pdf_to_string(str(app.config['UPLOAD_FOLDER'].joinpath(pdf_file)))
                patient = Patient()
                patient.generate_from_pdf(content)
                print(patient.type)
                x = predictor.predict(patient)*100
                os.remove(str(app.config['UPLOAD_FOLDER'].joinpath(pdf_file)))
                #DAVIDE TADY CONTENT JSOU TO NLP
                return render_estimation_template(x)
            else:
                return render_upload_template([0,resPDF,0,0])
        if "UploadXML" in request.form:
            resXML = process_file(request, "fileXML", "xml")
            if resXML == 0:
                xml_file = secure_filename(request.files["fileXML"].filename)
                xml_stream = StringIO()
                ret = parse_xml(str(app.config['UPLOAD_FOLDER'].joinpath(xml_file)), xml_stream)
                if ret == None:
                    return render_template("upload.html", result=[2,0,0,0])
                np_data = generate_numpy(xml_stream.getvalue())
                patient = Patient()
                patient.generate_from_ecg(np_data)
                print(patient.data)
                x = predictor.predict(patient)*100
                os.remove(str(app.config['UPLOAD_FOLDER'].joinpath(xml_file)))
                return render_estimation_template(x)
            else:
                return render_upload_template([resXML,0,0,0])
        if "UploadBOTH" in request.form:
            resPDF = process_file(request, "filePDF", "pdf")
            if resPDF == 0:
                resXML = process_file(request, "fileXML", "xml")
                if resXML == 0:
                    xml_stream = StringIO()
                    xml_file = secure_filename(request.files["fileXML"].filename)
                    pdf_file = secure_filename(request.files["filePDF"].filename)
                    ret = parse_xml(str(app.config['UPLOAD_FOLDER'].joinpath(xml_file)), xml_stream)
                    content = pdf_to_nlp.convert_pdf_to_string(str(app.config['UPLOAD_FOLDER'].joinpath(pdf_file)))
                    if ret == None:
                        return render_upload_template([2,0,0,0])
                    np_data = generate_numpy(xml_stream.getvalue())
                    patient = Patient()
                    patient.generate_from_pdf_and_ecg(content, np_data)
                    print(patient.data, patient.nlp)
                    x = predictor.predict(patient)*100
                    os.remove(str(app.config['UPLOAD_FOLDER'].joinpath(pdf_file)))
                    os.remove(str(app.config['UPLOAD_FOLDER'].joinpath(xml_file)))
                    return render_estimation_template(x)
                else:
                    return render_upload_template([0,0,0,resXML])
            else:
                return render_upload_template([0,0,resPDF,0])
    return render_upload_template([0,0,0,0])


def render_upload_template(arr):
    return render_template("upload.html", result=arr, errors=error_msgs(arr))

def render_estimation_template(estimation):
    return render_template("result.html", result=estimation)




def error_msgs(result): #[[0,0,0,0]]
    errors = [None, None, None, None]
    if result[0] == 1:
        errors[0] = "You first need to select a file!"
    if result[0] == 2:
        errors[0] = "You selected an invalid file!!"

    if result[1] == 1:
        errors[1] = "You first need to select a file!"
    if result[1] == 2:
        errors[1] = "You selected an invalid file!"
    if result[1] == 3:
        errors[1] = "Detection from medical report only is not yet implemented!"

    if result[2] == 1:
        errors[2] = "You first need to select a PDF file!"
    if result[2] == 2:
        errors[2] = "You selected an invalid PDF file!"

    if result[3] == 1:
        errors[2] = "You first need to select an XML file!"
    if result[3] == 2:
        errors[2] = "You selected an invalid XML file!"


    return errors



@app.route("/faq", methods=["GET", "POST"])
def display_faq():
    return render_template('faq.html')

@app.route("/about", methods=["GET", "POST"])
def display_about():
    return render_template('about.html')

def run(port=8000):
    app.run(debug=False, host='0.0.0.0', port=port)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=False)