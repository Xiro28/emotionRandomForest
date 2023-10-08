from forest import RandomForest
import joblib
import pathlib
from utils import write_to_file

import pandas as pd
import numpy as np
import cv2


TRAIN_PATH = "./archive/train/"
data_train_dir = pathlib.Path(TRAIN_PATH)

imagesData = []
imagesLabel = []

labelTextToNumber = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

def getImageLabel(path):
    #from before the last / to the last / (escluso)
    #esempio: ./archive/train/angry/0.jpg
    #ritorna angry
    return path.split("\\")[-2]

if __name__ == "__main__":
    print("Load dataset from path")


    # le immagini devono essere caricate e messe in un array nel seguente modo:
    # [indexImage, imageData (array reshaped 48*48)]
    # e deve corrispondere ad un label (ogni immagine ha un label)

    for dirs in data_train_dir.glob("*"):
  
        for photo in dirs.glob("*.*"):
            imgPath = str(photo)

            image = cv2.imread(imgPath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (48, 48))
            image = image.reshape(48*48)

            imagesLabel.append(labelTextToNumber[getImageLabel(imgPath)])
            imagesData.append(image)

    imagesData = np.array(imagesData)
    imagesLabel = np.array(imagesLabel)

    rf = RandomForest(max_depth=100 ,n_jobs=1)

    rf.fit(imagesData, imagesLabel)

    write_to_file(rf, "model.joblib")

