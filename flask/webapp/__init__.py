from flask import Flask
app = Flask(__name__)

from webapp import views
from pathlib import Path

UPLOAD_FOLDER = Path.home() / '/insight/FlaskDropSite'
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

