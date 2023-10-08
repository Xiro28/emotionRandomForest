import pathlib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import joblib

import cv2
import numpy as np
from skimage.feature import hog

from sklearn.ensemble import RandomForestClassifier  # Example algorithm choice

from PIL import Image


def getLabelFromPath(path):
    sPath = str(path)
    return sPath[sPath.rfind('/') + 1 : ]

labelTextToNum = {'angry':0,  'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad' : 5, 'surprise' : 6}
textToLabel = ['angry',  'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise' ]

#Prepare the data to be processed
data_train_dir = pathlib.Path("./archive/test/")

#print(load_iris())

imagesData = []

model = joblib.load("./emotional_face_modelV1_1.joblib")
labels = joblib.load("./emotional_face_modelV1_1_labels.joblib")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
       print("Can't receive frame (stream end?). Exiting ...")
       break
    
    #cropped_image = frame[50:, :330]
    resized = cv2.resize(frame, (48,48)) 
    imageData = np.array(resized)

    gray_frame = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
    
    gray_frame = gray_frame.reshape(-1, 48*48)
    
    

    #print(model.predict_proba(imageData))
    res = model.predict(gray_frame)
    
    print(textToLabel[res[0]])
    cv2.imshow('frame', gray_frame)
    cv2.waitKey(1)
            

