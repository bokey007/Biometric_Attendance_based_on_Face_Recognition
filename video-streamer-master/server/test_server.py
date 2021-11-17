from flask_socketio import SocketIO
from flask import Flask, render_template, request
from PIL import Image
from engineio.payload import Payload
from io import BytesIO
import base64
Payload.max_decode_packets = 500
import eventlet
eventlet.monkey_patch()



app = Flask(__name__)
socketio = SocketIO(app)




@app.route('/')
def entry_page():
    #return render_template('app_attendace_usermedia_socket.html')
    return render_template('test.html')


'''@app.route('/hello')
def hello():
    print("hello fuction that processes image is executed")
    #data_url = request.values['imageBase64']
    #data_url = data_url[22:]
    #print(data_url)
    #im = Image.open(BytesIO(base64.b64decode(data_url)))
    #print(type(im))
    #im.save('image.jpeg')
    return ''    '''


@socketio.on('connect', namespace='/web')
def connect_web():
    print('[INFO] Web client connected: {}'.format(request.sid))
    #socketio.emit('requiest_video', 'send_video', namespace='/web')


@socketio.on('disconnect', namespace='/web')
def disconnect_web():
    print('[INFO] Web client disconnected: {}'.format(request.sid))


'''@socketio.on('connect', namespace='/cv')
def connect_cv():
    print('[INFO] CV client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/cv')
def disconnect_cv():
    print('[INFO] CV client disconnected: {}'.format(request.sid))

    
@socketio.on('cv2server')
def handle_cv_message(message):
    socketio.emit('server2web', message, namespace='/web')
    #print('this is data from cv app to server')'''

@socketio.on('message', namespace='/web')
def handle_web_message(message):
    print('data received')
    print(message)
    #socketio.emit('server2cv', message, namespace='/cv')
    print('data emitted from server')


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001, debug=True)
