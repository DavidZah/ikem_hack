import io
from datetime import date
from flask import send_from_directory

today = date.today()
from flask import Flask, redirect, url_for, request, send_file, render_template



import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import random


UPLOAD_FOLDER = 'C:/Users/vkoro/ownCloud/HACKATHONGS/healthhack2021/ikem_hack/web/tmp'
pdf_file=""
xml_file=""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == extension

def do_ai_magic():
    return random.randrange(0,100)

@app.route('/uploads/<name>')
def download_file(name):
    
    return f'''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Jaká je šance že umřeš</h1>
    {x}
    '''

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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], pdf_file))
        return 0

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("what")
        if "UploadPDF" in request.form:
            # check if the post request has the file part
            resPDF = process_file(request, "filePDF", "pdf")
            if resPDF == 0:
                x = do_ai_magic()
                return render_template("dead.html", result=["width:"+str(x)+"%", str(x)])
            else:
                return render_template("upload.html", result=[0,resPDF,0,0])
        if "UploadXML" in request.form:
            resXML = process_file(request, "fileXML", "xml")
            if resXML == 0:
                x = do_ai_magic()
                return render_template("dead.html", result=["width:"+str(x)+"%", str(x)])
            else:
                return render_template("upload.html", result=[resXML,0,0,0])
        if "UploadBOTH" in request.form:
            resPDF = process_file(request, "filePDF", "pdf")
            if resPDF == 0:
                resXML = process_file(request, "fileXML", "xml")
                if resXML == 0:
                    x = do_ai_magic()
                    return render_template("dead.html", result=["width:"+str(x)+"%", str(x)])
                else:
                    return render_template("upload.html", result=[0,0,0,resXML])
            else:
                return render_template("upload.html", result=[0,0,resPDF,0])
    return render_template('upload.html', result=[0,0,0,0])

@app.route('/xmlresult', methods=['GET', 'POST'])
def upload_xml():
    pass

@app.route('/preciseresult', methods=['GET', 'POST'])
def upload_precise():
    pass





def run(port=8000):
    app.run(debug=False, host='0.0.0.0', port=port)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=False)