from collections import UserDict
import os, sys

from numpy import broadcast
from workspace._global.config import * 
from flask import Flask, jsonify, request
from workspace.api_funcs.add_face import new_face_added, detect_face
import tensorflow as tf
from tensorflow import keras
from object_detection import * 
from models.research.object_detection.utils import label_map_util
from flask_socketio import SocketIO, emit, disconnect
from google.protobuf.json_format import MessageToJson

app = Flask(__name__)
model = keras.models.load_model(OUTPUT_FACES_RECOG)
label_map = label_map_util.load_labelmap(FACES_RECOG_LABEL_MAP)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/add', methods=['POST'])
def add_user():

    user = request.get_json()
    if not user:
        return
    
    new_model, classes_cnt = new_face_added(user['name'], user['dir'])
    global model
    model = new_model

    global label_map
    label_map = label_map_util.load_labelmap(FACES_RECOG_LABEL_MAP)
    # categories.append(
    #     { "id": classes_cnt user['name']}
    # )


    emit('refetch', MessageToJson(label_map), broadcast=True, namespace='')

    return jsonify({"Updated model": "hello world"})


@socketio.on('connected')
def handle_connected():
    emit('refetch', MessageToJson(label_map))


@app.route('/detect', methods=['POST'])
def detect():
    face = request.get_json()
    if not face:
        return

    label, score = detect_face(model, label_map.item, face['path'])
    return jsonify({
        "label": label,
        "score": str(score)
    })



@app.after_request
def after_request_func(response):
    origin = request.headers.get('Origin')
    if request.method == 'OPTIONS':
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Headers', 'x-csrf-token')
        response.headers.add('Access-Control-Allow-Methods',
                            'GET, POST, OPTIONS, PUT, PATCH, DELETE')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
    else:
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)

    return response
### end CORS section

if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app, debug=True, host="localhost")