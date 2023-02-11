import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import speech_recognition as sr
import pyttsx3
import time

engine = pyttsx3.init('sapi5') #Microsoft speak api module
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice',voices[1].id)  # 0 for male voice and 1 for female voice 

def speak(audio):   
    ''' This function is used to speak '''
    engine.say(audio)
    engine.runAndWait()


cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2)  #Setting the maximum hands

prediction =  None
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
labels = ["1", "2" , "3", "4", "5", "6", "7" , "8" , "9", "A", "B" , "C" , "D", "E", "F", "G", "H" , "I", 'J','K',"L", "M" , "N","O", "P", "Q", "R", "S","T","U", "V","W","X","Y","Z" ]
offset = 20
imgSize = 350

result = []


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        # classifier = Classifier("Model/keras_model_one_hand.h5","Model/labels_one_hand.txt")
        # labels = ["1", "2" , "3", "4", "5", "6", "7" , "8" , "9", "C" , "I", "L", "O", "U", "V"]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
        # Change The Slicing According The Requirement
        imgCrop = img[y-offset-10: y+h+offset+120,x-offset-10:x+w+offset+250]# imgCrop = img[10: y+h+offset+100,10:x+w+offset+200]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        # if aspectRatio>1:
        #     k= imgSize/h
        #     wCal = math.ceil(k*w)
        #     imgResize=cv2.resize(imgCrop,(wCal,imgSize))
        #     imgResizeShape = imgResize.shape
        #     wGap = math.ceil((imgSize - wCal)/2)
        #     imgWhite[:,wGap:wCal + wGap] = imgResize
        #     prediction,index = classifier.getPrediction(imgWhite,draw=False)
        #     print(prediction,index)
        
        # else:
        #     k = imgSize/w
        #     hCal = math.ceil(k*h)
        #     imgResize = cv2.resize(imgCrop,(imgSize,hCal))
        #     imgResizeShape = imgResize.shape
        #     hGap = math.ceil((imgSize-hCal)/2)
        #     imgWhite[hGap:hCal+hGap,:]=imgResize
        #     prediction,index = classifier.getPrediction(imgWhite,draw=False)
            # print(prediction,index)

        prediction,index = classifier.getPrediction(img)
        # print(prediction,index)
        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4) 
        # speak(labels[index]) 

        # cv2.imshow("ImageCrop",imgCrop)
        # cv2.imshow("ImageWhite",imgWhite)


    elif len(hands)==2:
        off = 20
        hand = hands[1]
        # classifier = Classifier("Model/keras_model_two_hand.h5","Model/labels_two_hand.txt")
        # labels = ["A", "B" , "D", "E", "F", "G", "H" , "J" , "K", "M" , "N", "P", "Q", "R", "S","T","W","X","Y","Z"]
        x,y,w,h = hand['bbox'] #BOunding Box information

        imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
        # Change The Slicing According The Requirement
        imgCrop = img[10: y+h+off+120,10:x+w+off+250]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        # if aspectRatio>1:
        #     k= imgSize/h
        #     wCal = math.ceil(k*w)
        #     imgResize=cv2.resize(imgCrop,(wCal,imgSize))
        #     imgResizeShape = imgResize.shape
        #     wGap = math.ceil((imgSize - wCal)/2)
        #     imgWhite[:,wGap:wCal + wGap] = imgResize
        #     prediction,index = classifier.getPrediction(imgWhite,draw=False)
        #     print(prediction,index)
        
        # else:
        #     k = imgSize/w
        #     hCal = math.ceil(k*h)
        #     imgResize = cv2.resize(imgCrop,(imgSize,hCal))
        #     imgResizeShape = imgResize.shape
        #     hGap = math.ceil((imgSize-hCal)/2)
        #     imgWhite[hGap:hCal+hGap,:]=imgResize
        #     prediction,index = classifier.getPrediction(imgWhite,draw=False)
        #     # print(prediction,index)

        prediction,index = classifier.getPrediction(img)
        # print(prediction,index)
        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        # speak(labels[index])

        # cv2.imshow("ImageCrop",imgCrop)
        # cv2.imshow("ImageWhite",imgWhite)

          

    cv2.imshow("Image",imgOutput)
    key = cv2.waitKey(1)

    if len(hands) != 0 :
        for i in prediction:
            if i > 0.1 and i <= 1.0:
                time.sleep(5)
                result.append(labels[index])
                    
    if key==ord('e'):
        break

final_result = "".join([str(i) for i in result])
print(final_result)
speak(final_result)