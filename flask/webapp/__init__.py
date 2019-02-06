import os
import binascii

from flask import Flask
app = Flask(__name__)
app.secret_key = binascii.hexlify(os.urandom(24))

from webapp import views
from pathlib import Path

UPLOAD_FOLDER = Path.home() / 'insight/FlaskDropSite'
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER