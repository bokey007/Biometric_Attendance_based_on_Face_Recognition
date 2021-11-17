from flask_socketio import SocketIO
from flask import Flask, render_template, request
from engineio.payload import Payload
#Payload.max_decode_packets = 500
import eventlet
eventlet.monkey_patch()

import argparse
import base64
import imutils
import pickle
import io
import cv2
from PIL import Image
import numpy as np
import face_recognition
from datetime import datetime
import time
from pytz import timezone
import openpyxl



parser = argparse.ArgumentParser(description='attendance system')
parser.add_argument(
    '--camera', type=int, default='0',
    help='The camera index to stream from.')
parser.add_argument(
    '--use-streamer', action='store_true',
    help='Use the embedded streamer instead of connecting to the server.')
parser.add_argument(
    '--server-addr', type=str, default='localhost',
    help='The IP address or hostname of the SocketIO server.')
parser.add_argument(
    '--stream-fps', type=float, default=20.0,
    help='The rate to send frames to the server.')
parser.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
parser.add_argument("-d", "--detection-method", type=str, default="hog",
                    help="face detection model to use: either `hog` or `cnn`")
args = parser.parse_args()

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args.encodings, "rb").read())

# load knn classifier
print('[info] loading knn classifire')
with open('knn_classifire.pickle', 'rb') as f:
    knn_clf = pickle.load(f)

app = Flask(__name__)
socketio = SocketIO(app)


def convert_image_to_jpeg(image):
    # Encode frame as jpeg
    frame = cv2.imencode('.jpg', image)[1].tobytes()
    # Encode frame in base64 representation and remove
    # utf-8 encoding
    frame = base64.b64encode(frame).decode('utf-8')
    return "data:image/jpeg;base64,{}".format(frame)


@app.route('/')
def index():
    """Home page."""
    return render_template('base.html')

@app.route('/entry_page')
def entry_page():
    #return render_template('app_attendace_usermedia_socket.html')
    return render_template('app_attendace_usermedia_socket.html')


@socketio.on('connect', namespace='/web')
def connect_web():
    print('[INFO] Web client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/web')
def disconnect_web():
    print('[INFO] Web client disconnected: {}'.format(request.sid))


@socketio.on('connect', namespace='/cv')
def connect_cv():
    print('[INFO] CV client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/cv')
def disconnect_cv():
    print('[INFO] CV client disconnected: {}'.format(request.sid))


@socketio.on('cv2server')
def handle_cv_message(message):
    socketio.emit('server2web', message, namespace='/web')
    #print('this is data from cv app to server')

@socketio.on('message', namespace='/web')
def handle_web_message(message):

    print('data received')
    b = io.BytesIO(base64.b64decode(message))
    pimg = Image.open(b)
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    pimg.close()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb, model=args.detection_method)
    encodings_face = face_recognition.face_encodings(rgb, boxes)

    names = []
    text = ["Please come closer to the camera"]

    # loop over the facial embeddings
    for encoding in encodings_face:
        # attempt to match each face in the input image to our known
        # encodings
        name = "Unknown"
        # print(len(encoding))
        # print(type(encoding))
        closest_distances, closest_index = knn_clf.kneighbors([encoding], n_neighbors=1)
        print('closest dist is ')
        print(closest_distances[0][0])
        # distance threshold
        print('closest idx is')
        # print(closest_index[0][0])
        print(closest_index[0][0])
        # print(type(closest_index))
        if closest_distances[0][0] <= 0.4:
            name = data["names"][closest_index[0][0]]
        else:
            name = "Unknown"
        # update the list of names
        names.append(name)
        name = "Unknown"

    # acquire the lock, set the output frame, and release the
    # lock
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        last_name = name
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        # ensure the face width and height are sufficiently large
        if bottom - top > 180 or right - left > 180:
            # Generate text to display on streamer
            text = ["Hey {}! your entry is logged".format(name)]
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name + ' Entry logged', (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
            msg1 = "Hey "
            msg2 = name
            msg3 = "! your entry is logged."
            msg_f = msg1 + msg2 + msg3
            print(msg_f)
            wb = openpyxl.load_workbook('attendance.xlsx')
            active_sheet = wb['entry']
            india = timezone('Asia/Kolkata')
            # date = str(datetime.now(india))[:10] + "@" + str(datetime.now())[11:16] + "hrs"
            date = str(datetime.now(india))[:10]
            entry_time = str(datetime.now())[11:16] + "hrs"
            row = (name, date, entry_time)
            print(row)
            active_sheet.append(row)
            wb.save('attendance.xlsx')
            print('excel sheet updated')
            frame = frame.copy()
        else:
            # Generate text to display on streamer
            text = ["Please come closer to the camera"]

    socketio.emit('server2web',
                  {
                    'image': convert_image_to_jpeg(frame),
                    'text': '<br />'.join(text)
                   }, namespace='/web')



    #print(message)
    #socketio.emit('server2cv', message, namespace='/cv')
    print('data emitted from server')


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001, debug=True)
