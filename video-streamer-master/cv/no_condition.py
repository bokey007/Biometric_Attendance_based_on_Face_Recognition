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
        self.server_port = 5001
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
    # obj_detect = edgeiq.ObjectDetection(
    #        "alwaysai/mobilenet_ssd")
    # obj_detect.load(engine=edgeiq.Engine.DNN)

    # print("Loaded model:\n{}\n".format(obj_detect.model_id))
    # print("Engine: {}".format(obj_detect.engine))
    # print("Accelerator: {}\n".format(obj_detect.accelerator))
    # print("Labels:\n{}\n".format(obj_detect.labels))
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
        if use_streamer:
            streamer = edgeiq.Streamer().setup()
        else:
            streamer = CVClient(server_addr, stream_fps).setup()

        with edgeiq.WebcamVideoStream(cam=camera) as video_stream:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:

                global names
                names = []
                global max_value
                max_value = 0
                global boxes
                boxes = []

                global max_key
                max_key = None
                global msg_f
                msg_f = None

                while True:
                    frame = video_stream.read()

                    # convert the input frame from BGR to RGB then resize it to have
                    # a width of 750px (to speedup processing)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = imutils.resize(frame, width=750)
                    r = frame.shape[1] / float(rgb.shape[1])

                    # detect the (x, y)-coordinates of the bounding boxes
                    # corresponding to each face in the input frame, then compute
                    # the facial embeddings for each face
                    boxes = face_recognition.face_locations(rgb, model=detection_method)

                    encodings_face = face_recognition.face_encodings(rgb, boxes)

                    # loop over the facial embeddings
                    for encoding in encodings_face:
                        # attempt to match each face in the input image to our known
                        # encodings
                        name = "Unknown"
                        # print(len(encoding))
                        # print(type(encoding))
                        """
                        predict_proba = knn_clf.predict_proba([encoding]).tolist()
                        prob_max = max(predict_proba[0])
                        prob_max_index = predict_proba[0].index(prob_max)
                        #print("prob start")
                        #print(predict_proba)
                        #print("prob end")

                        if prob_max >= 0.99:
                            name = known_identities[prob_max_index]
                        else:
                            name = "Unknown"

                            """

                        closest_distances, closest_index = knn_clf.kneighbors([encoding], n_neighbors=1)
                        print('closest dist is ')
                        print(closest_distances[0][0])
                        # distance threshold
                        print('closest idx is')
                        # print(closest_index[0][0])
                        print(closest_index[0][0])
                        # print(type(closest_index))
                        if closest_distances[0][0] <= 0.45:
                            name = data["names"][closest_index[0][0]]
                        else:
                            name = "Unknown"

                        """"
                        matches = face_recognition.compare_faces(data["encodings"], encoding)
                        name = "Unknown"

                        # check to see if we have found a match
                        if True in matches:
                            # find the indexes of all matched faces then initialize a
                            # dictionary to count the total number of times each face
                            # was matched
                            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                            counts = {}

                            # loop over the matched indexes and maintain a count for
                            # each recognized face face
                            for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1

                            # determine the recognized face with the largest number
                            # of votes (note: in the event of an unlikely tie Python
                            # will select first entry in the dictionary)
                            name = max(counts, key=counts.get) """

                        # update the list of names
                        names.append(name)

                        # loop over the recognized faces
                        for ((top, right, bottom, left), name) in zip(boxes, names):
                            # rescale the face coordinates
                            top = int(top * r)
                            right = int(right * r)
                            bottom = int(bottom * r)
                            left = int(left * r)

                            # ensure the face width and height are sufficiently large
                            if bottom - top < 180 or right - left < 180:
                                continue

                            # draw the predicted face name on the image
                            cv2.rectangle(frame, (left, top), (right, bottom),
                                          (0, 255, 0), 2)
                            y = top - 15 if top - 15 > 15 else top + 15
                            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75, (0, 255, 0), 2)
                        name = "Unknown"
                        # name_buffer.append(name)
                    if len(names) > 0:
                        max_key, max_value = freq(names)
                        print(max_value, max_key)

                    if max_value > 1:
                        break



                    msg1 = "Hey "
                    msg2 = str(max_key)
                    print(max_key)
                    msg3 = "! your entry is logged."
                    msg_f = msg1 + msg2 + msg3
                    print(msg_f)
                    wb = openpyxl.load_workbook('attendance.xlsx')
                    active_sheet = wb['entry']
                    india = timezone('Asia/Kolkata')
                    date = str(datetime.now(india))[:10] + "@" + str(datetime.now())[11:16] + "hrs"
                    row = (max_key, date)
                    print(row)
                    active_sheet.append(row)
                    wb.save('attendance.xlsx')

                frame = frame.copy()

                # results = obj_detect.detect_objects(frame, confidence_level=.5)
                # frame = edgeiq.markup_image(
                #        frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Hey {}! your entry is logged".format(max_key)]
                # text.append(
                #        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                # for prediction in results.predictions:
                #    text.append("{}: {:2.2f}%".format(
                #        prediction.label, prediction.confidence * 100))

                streamer.send_data(frame, text)

                fps.update()

                max_key = None
                max_value = 0

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
    parser.add_argument("-d", "--detection-method", type=str, default="cnn",
                        help="face detection model to use: either `hog` or `cnn`")
    args = parser.parse_args()
    main(args.camera, args.use_streamer, args.server_addr, args.stream_fps, args.encodings, args.detection_method)
