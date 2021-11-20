import io
from datetime import date
from flask import send_from_directory

today = date.today()
from flask import Flask, redirect, url_for, request, send_file, render_template



import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import random


<<<<<<< Updated upstream
UPLOAD_FOLDER = 'tmp\\'
ALLOWED_EXTENSIONS = {'xml', 'pdf'}
=======
UPLOAD_FOLDER = 'C:/Users/vkoro/ownCloud/HACKATHONGS/healthhack2021/ikem_hack/web/tmp/'

pdf_file = ""
xml_file = ""
>>>>>>> Stashed changes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == extension

def do_ai_magic():
    return random.randrange(0,100)/100

@app.route('/uploads/<name>')
def download_file(name):
    x = do_ai_magic()
    return f'''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Jaká je šance že umřeš</h1>
    {x}
    '''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return render_template('upload.html')

@app.route('/pdfresult', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return render_template('upload.html')

@app.route('/xmlresult', methods=['GET', 'POST'])
def upload_xml():
    pass

@app.route('/preciseresult', methods=['GET', 'POST'])
def upload_precise():
    pass





def run(port=8000):
    app.run(debug=False, host='0.0.0.0', port=port)


if __name__ == '__main__':
    app.run(debug=False)