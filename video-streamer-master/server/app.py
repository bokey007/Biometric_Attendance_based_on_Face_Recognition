from flask_socketio import SocketIO
from flask import Flask, render_template, request
from flask import Blueprint, render_template, request, redirect, flash, url_for
#from flask_login import login_required, current_user
#from werkzeug.utils import secure_filename
#import os
#from . import app
#from . import db

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    """Home page."""
    return render_template('base.html')


@app.route('/entry_page')
def entry_page():
    return render_template('drishti_indexresent2_base.html')

@socketio.on('connect', namespace='/')
def connect_web():
    print('[INFO] home page client connected: {}'.format(request.sid))

@socketio.on('disconnect', namespace='/')
def disconnect_web():
    print('[INFO] home page client disconnected: {}'.format(request.sid))


@socketio.on('connect', namespace='/web')
def connect_web():
    print('[INFO] face Web client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/web')
def disconnect_web():
    print('[INFO] face Web client disconnected: {}'.format(request.sid))


@socketio.on('connect', namespace='/cv')
def connect_cv():
    print('[INFO] CV client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/cv')
def disconnect_cv():
    print('[INFO] CV client disconnected: {}'.format(request.sid))


@socketio.on('cv2server')
def handle_cv_message(message):
    socketio.emit('server2web', message, namespace='/web')
    #print(type(message['text']))
    #print(message['text'])
    #print(message)


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001)