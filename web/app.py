from datetime import date

today = date.today()
from flask import render_template

from flask import Flask, flash, request
from werkzeug.utils import secure_filename
import random
from dataset_use import Predictor
from utils import *
from patient import Patient
import pdf_to_nlp


UPLOAD_FOLDER = Path("tmp")
pdf_file=""
xml_file=""
json_data = None

xml_stream = StringIO()
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
                print(patient.nlp)
                x = do_ai_magic()
                #DAVIDE TADY CONTENT JSOU TO NLP
                return render_template("dead.html", result=["width:"+str(x)+"%", str(x)])
            else:
                return render_template("upload.html", result=[0,resPDF,0,0])
        if "UploadXML" in request.form:
            resXML = process_file(request, "fileXML", "xml")
            if resXML == 0:
                xml_file = secure_filename(request.files["fileXML"].filename)
                ret = parse_xml(str(app.config['UPLOAD_FOLDER'].joinpath(xml_file)), xml_stream)
                if ret == None:
                    return render_template("upload.html", result=[2,0,0,0])
                np_data = generate_numpy(xml_stream.getvalue())
                patient = Patient()
                patient.generate_from_ecg(np_data)
                print(patient.data)
                x = predictor.predict(patient)*100
                xml_stream.truncate(0)
                return render_template("dead.html", result=["width:"+str(x)+"%", str(x)])
            else:
                return render_template("upload.html", result=[resXML,0,0,0])
        if "UploadBOTH" in request.form:
            resPDF = process_file(request, "filePDF", "pdf")
            if resPDF == 0:
                resXML = process_file(request, "fileXML", "xml")
                if resXML == 0:
                    xml_file = secure_filename(request.files["fileXML"].filename)
                    pdf_file = secure_filename(request.files["filePDF"].filename)
                    ret = parse_xml(str(app.config['UPLOAD_FOLDER'].joinpath(xml_file)), xml_stream)
                    content = pdf_to_nlp.convert_pdf_to_string(str(app.config['UPLOAD_FOLDER'].joinpath(pdf_file)))
                    if ret == None:
                        return render_template("upload.html", result=[2,0,0,0])
                    np_data = generate_numpy(xml_stream.getvalue())
                    patient = Patient()
                    patient.generate_from_pdf_and_ecg(content, np_data)
                    print(patient.data, patient.nlp)
                    x = predictor.predict(patient)*100
                    xml_stream.truncate(0)
                    return render_template("dead.html", result=["width:"+str(x)+"%", str(x)])
                else:
                    return render_template("upload.html", result=[0,0,0,resXML])
            else:
                return render_template("upload.html", result=[0,0,resPDF,0])
    return render_template('upload.html', result=[0,0,0,0])

def run(port=8000):
    app.run(debug=False, host='0.0.0.0', port=port)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=False)