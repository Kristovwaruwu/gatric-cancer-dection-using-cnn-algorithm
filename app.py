from flask import Flask, jsonify, request, render_template, send_file, make_response
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from Api import Api
import sys
import os
import time
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

obj = Api()

UPLOAD_FOLDER = "input/tempData/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    path = request.path
    return render_template('home.html', data=path)


@app.route('/process', methods=['POST'])
def detection_process():
    path = request.path

    error = ""

    if 'file' in request.files:
        filetxt = request.files["file"]
        if filetxt and allowed_file(filetxt.filename):
            filename = secure_filename(filetxt.filename)
            print(filename, filetxt.filename)
            filetxt.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            error = "Format file salah"

    img = obj.read_img('input/tempData/' + filename)
    img_tensor = obj.load_image('input/tempData/' + filename)

    obj.save_img(img, 'gambarHasil.'+filename.split('.')[-1])
    model = obj.loadModel('model_new.h5', img_tensor)
    classes = obj.predict(model, img_tensor)
    try:
        return jsonify({'code': 200, 'message': 'Success', 'data': classes}), 200
    except error:
        return jsonify({'code': 500, 'message': 'Error', 'error': str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8080)
