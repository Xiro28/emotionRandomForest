import joblib
import pathlib

from pyforest.forest import RandomForest

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import numpy as np


TRAIN_PATH = "./archive/train/"
data_train_dir = pathlib.Path(TRAIN_PATH)

imagesPath = []

print("Load dataset from path")

for photo in data_train_dir.glob("*.*"):
    imagesPath.append(photo)
    
train_datagen = ImageDataGenerator(rescale = 1./255)
train_y = train_datagen.flow_from_directory(TRAIN_PATH,
                                                 target_size = (48, 48),
                                                 batch_size = 32,
                                                 class_mode = 'sparse').classes

print("Start training")

rf = RandomForest()
rf.fit(imagesPath, train_y)


print("DUMP")

joblib.dump(model, "./emotional_face_modelV1_2.joblib")
joblib.dump(train_y, "./emotional_face_modelV1_1_labels.joblib")
