from webapp import app
from flask import render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from pathlib import Path
import sys
sys.path.append('../')
from auto_anonymize.anonymize_vid import anonymize_vid


ALLOWED_EXTENSIONS = set(['mp4', 'MOV'])
RECOGNIZED_FACES_DIR = Path(__file__).parents[2] / 'data/faces_db/'
assert RECOGNIZED_FACES_DIR.is_dir(), \
    'RECOGNIZED_FACES_DIR does not exist.'
RECOGNIZED_FACES_DIR = str(RECOGNIZED_FACES_DIR)


def valid_file(filename):
    return '.' in filename and \
        filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS 


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure post request contains file attr
        if 'file' not in request.files:
            flash('[ERROR] Post object does not contain file attribute.')
            return redirect(request.url)

        file = request.files['file']

        # Ensure file is not empty
        if file.filename == '':
            flash('No file was provided.')
            return redirect(request.url)

        # Reads file
        if file and valid_file(file.filename):
            filename = secure_filename(file.filename)
            mod_filename = '_mod.'.join(filename.rsplit('.', 1))
            src = str(Path(app.config['UPLOAD_FOLDER']) / filename)

            file.save(src)
            anonymize_vid(src, known_faces_loc=RECOGNIZED_FACES_DIR, 
                          use_retinanet=False)

            return redirect(url_for('uploaded_file', filename=mod_filename))

    return render_template('index.html', title='Home')


@app.route('/downloads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)