from sklearn import neighbors

from flask import Flask,render_template,request,redirect,url_for,session
import MySQLdb
import os

#import for face recongition
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.face_recognition_cli import image_files_in_folder

n_neighbors = 5
# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open('encodings2.pickle', "rb").read())

if n_neighbors is None:
    n_neighbors = int(round(sqrt(len(data["encodings"]))))
    print("Chose n_neighbors automatically as:", n_neighbors)

knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
knn_clf.fit(data["encodings"], data["names"])


with open('knn_classifire.pickle', 'wb') as f:
     pickle.dump(knn_clf, f)

print("classifire trained")