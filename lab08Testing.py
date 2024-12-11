import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tensorflow import keras

# Método para cargar labels personalizdo
def cargar_labels(archivo):
    labels = []
    try:
        with open(archivo, "r") as f:
            for linea in f:
                indice, valor = linea.strip().split(" ", 1)
                labels.append(valor)
        return labels
    except FileNotFoundError:
        print(f"El archivo {archivo} no fue encontrado.")
        return []
    
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("ModelS/keras_model.h5", "ModelS/labels.txt")
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
string = "Inicia con la tecla z"
# labels = ["A", "B", "C", "D", "E", "ESP"]
archivo_labels = "ModelS/labels.txt"
labels = cargar_labels(archivo_labels)
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        #Texto Prueba
        cv2.putText(imgOutput, string, (00, 185), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(imgOutput, "Borra con la z, y con s colocas un nuevo caracter", (00, 300), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("z"):
        string = ""
    if key == ord("s"):
        if labels[index] == "ESP":
            string = string + " "
        else:
            string = string + labels[index]