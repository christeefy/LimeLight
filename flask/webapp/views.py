from webapp import app
from flask import render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import time
import hashlib
import warnings

from pathlib import Path
import sys
sys.path.append('../')
from limelight.anonymize_vid import anonymize_vid


def valid_file(filename, mode):
    '''
    Checks whether file is a valid file type.

    Inputs:
        filename: A string
        mode: Valid values include ['img', 'vid'].
    '''
    ALLOWED_EXTENSIONS = {
        'vid': set(['mp4', 'MOV']), 
        'img': set(['jpg'])
    }

    assert mode in ALLOWED_EXTENSIONS.keys(), 'Invalid mode provided.'
    return '.' in filename and \
        filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS[mode]


def rand_hash(length=8):
    '''
    Generate a random hex hash value of `length`.
    '''
    time_str = str(time.time()).encode('utf-8')
    return hashlib.md5(time_str).hexdigest()[:length]


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure post request contains a video file
        if 'video' not in request.files:
            warnings.warn('[ERROR] Post object does not contain file attribute.')
            return redirect(request.url)

        video = request.files['video']

        # Ensure file is not empty
        if video.filename == '':
            warnings.warn('No file was provided.')
            return redirect(request.url)

        # Reads file
        if video and valid_file(video.filename, 'vid'):
            filename = secure_filename(video.filename)
            mod_filename = '_mod.'.join(filename.rsplit('.', 1))

            # Save data into a unique session folder for each job
            session_id = rand_hash()
            upload_dst = app.config['UPLOAD_FOLDER'] / session_id
            upload_dst.mkdir(parents=True)
            src = str(upload_dst / filename)
            video.save(src)
            
            # Save faces if they exist
            if 'faces' in request.files:
                faces = request.files.getlist('faces')
                for face in faces:
                    assert valid_file(face.filename, 'img'), \
                        'Unsupported file format. Only JPG images allowed.'
                    face.save(str(upload_dst / secure_filename(face.filename)))

            # API to anonymize video
            anonymize_vid(src, known_faces_loc=upload_dst if 'faces' in request.files else None, 
                          use_retinanet=True, threshold=0.5, mark_faces=True)

            return redirect(url_for('uploaded_file', filename=mod_filename, sess_id=session_id))
    return render_template('index.html', title='Home')


@app.route('/downloads/<filename>&<sess_id>')
def uploaded_file(filename, sess_id=None):
    return send_from_directory(app.config['UPLOAD_FOLDER'] / sess_id, filename, 
                               as_attachment=True)
