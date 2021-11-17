import edgeiq
import socketio
import base64

from datetime import datetime

import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

from pytz import timezone

import openpyxl
from sklearn import neighbors

"""
Use object detection to detect objects in the frame in realtime. The
types of objects detected can be changed by selecting different models.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""

sio = socketio.Client()

max_value = 0
max_key = None
names = []
msg_f = None
known_identities = None
boxes = []


# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(src=0).start()
# time.sleep(2.0)


@sio.event
def connect():
    print('[INFO] Successfully connected to server.')


@sio.event
def connect_error():
    print('[INFO] Failed to connect to server.')


@sio.event
def disconnect():
    print('[INFO] Disconnected from server.')


class CVClient(object):
    def __init__(self, server_addr, stream_fps):
        self.server_addr = server_addr
        self.server_port = 5002
        self._stream_fps = stream_fps
        self._last_update_t = time.time()
        self._wait_t = (1 / self._stream_fps)

    def setup(self):
        print('[INFO] Connecting to server http://{}:{}...'.format(
            self.server_addr, self.server_port))
        sio.connect(
            'http://{}:{}'.format(self.server_addr, self.server_port),
            transports=['websocket'],
            namespaces=['/cv'])
        time.sleep(1)
        return self

    def _convert_image_to_jpeg(self, image):
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, text):
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            frame = edgeiq.resize(
                frame, width=640, height=480, keep_scale=True)
            sio.emit(
                'cv2server',
                {
                    'image': self._convert_image_to_jpeg(frame),
                    'text': '<br />'.join(text)
                })

    def check_exit(self):
        pass

    def close(self):
        sio.disconnect()


def freq(list):
    frequency = {}
    for item in list:
        frequency[item] = list.count(item)
    max_key1 = max(frequency, key=frequency.get)
    all_values = frequency.values()
    max_value1 = max(all_values)
    return max_key1, max_value1


def main(camera, use_streamer, server_addr, stream_fps, encodings, detection_method):

    known_identities_file = open("names_of_known_identities.txt", "r")
    global known_identities
    known_identities = known_identities_file.read().splitlines()

    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open(encodings, "rb").read())

    # load knn classifier
    print('[info] loading knn classifire')
    with open('knn_classifire.pickle', 'rb') as f:
        knn_clf = pickle.load(f)

    fps = edgeiq.FPS()

    try:
        streamer = None
        streamer = CVClient(server_addr, stream_fps).setup()

        with edgeiq.WebcamVideoStream(cam=camera) as video_stream:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:

                global names
                names = []

                global last_name
                last_name = ''
                global boxes
                boxes = []

                global msg_f
                msg_f = None

                global text
                text= []


                frame = video_stream.read()
                # convert the input frame from BGR to RGB then resize it to have
                # a width of 750px (to speedup processing)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(rgb, width=750)
                r = frame.shape[1] / float(rgb.shape[1])
                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input frame, then compute
                # the facial embeddings for each face
                boxes = face_recognition.face_locations(rgb, model=detection_method)
                encodings_face = face_recognition.face_encodings(rgb, boxes)
                if len(boxes) == 0:
                    text = ["Please come closer or Uncover your face"]
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
                    # name_buffer.append(name)
                # streamer.send_data(frame, 'Please come closer')
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
                        text = ["Hey {}! your exit is logged".format(name)]

                        # draw the predicted face name on the image
                        cv2.rectangle(frame, (left, top), (right, bottom),
                                      (0, 255, 0), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(frame, name + ' exit logged', (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (0, 255, 0), 2)

                        msg1 = "Hey "
                        msg2 = name
                        print(max_key)
                        msg3 = "! your exit is logged."
                        msg_f = msg1 + msg2 + msg3
                        print(msg_f)

                        wb = openpyxl.load_workbook('attendance_exit.xlsx')
                        active_sheet = wb['exit']
                        india = timezone('Asia/Kolkata')
                        #date = str(datetime.now(india))[:10] + "@" + str(datetime.now())[11:16] + "hrs"
                        date = str(datetime.now(india))[:10]
                        exit_time = str(datetime.now())[11:16] + "hrs"
                        row = (name, date, exit_time)
                        print(row)
                        active_sheet.append(row)
                        wb.save('attendance_exit.xlsx')
                        print('excel sheet updated')
                        frame = frame.copy()


                    else:
                        # Generate text to display on streamer
                        text = ["Please come closer to the camera"]

                #frame = frame.copy()


                # text.append(
                #        "Inference time: {:1.3f} s".format(results.duration))
                #text.append("Objects:")

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        if streamer is not None:
            streamer.close()
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='alwaysAI Video Streamer')
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
    main(args.camera, args.use_streamer, args.server_addr, args.stream_fps, args.encodings, args.detection_method)
