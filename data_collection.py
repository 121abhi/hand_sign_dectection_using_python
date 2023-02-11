import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  #Setting the maximum hands

offset = 20
imgSize = 350
folder = "Data/Z"
counter =0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
        # Change The Slicing According The Requirement
        imgCrop = img[y-offset-10: y+h+offset+120,x-offset-10:x+w+offset+250]# imgCrop = img[10: y+h+offset+100,10:x+w+offset+200]

        imgCropShape = imgCrop.shape

        # aspectRatio = h/w

        # if aspectRatio>1:
        #     k= imgSize/h
        #     wCal = math.ceil(k*w)
        #     imgResize=cv2.resize(imgCrop,(wCal,imgSize))
        #     imgResizeShape = imgResize.shape
        #     wGap = math.ceil((imgSize - wCal)/2)
        #     imgWhite[:,wGap:wCal + wGap] = imgResize
        
        # else:
        #     k = imgSize/w
        #     hCal = math.ceil(k*h)
        #     imgResize = cv2.resize(imgCrop,(imgSize,hCal))
        #     imgResizeShape = imgResize.shape
        #     hGap = math.ceil((imgSize-hCal)/2)
        #     imgWhite[hGap:hCal+hGap,:]=imgResize

        # cv2.imshow("ImageCrop",imgCrop)
        # cv2.imshow("ImageWhite",imgWhite)

    elif len(hands)==2:
        off = 20
        hand = hands[1]
        x,y,w,h = hand['bbox'] #BOunding Box information

        imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
        # Change The Slicing According The Requirement
        imgCrop = img[10: y+h+off+120,10:x+w+off+250]

        imgCropShape = imgCrop.shape

        # aspectRatio = h/w

        # if aspectRatio>1:
        #     k= imgSize/h
        #     wCal = math.ceil(k*w)
        #     imgResize=cv2.resize(imgCrop,(wCal,imgSize))
        #     imgResizeShape = imgResize.shape
        #     wGap = math.ceil((imgSize - wCal)/2)
        #     imgWhite[:,wGap:wCal + wGap] = imgResize
        
        # else:
        #     k = imgSize/w
        #     hCal = math.ceil(k*h)
        #     imgResize = cv2.resize(imgCrop,(imgSize,hCal))
        #     imgResizeShape = imgResize.shape
        #     hGap = math.ceil((imgSize-hCal)/2)
        #     imgWhite[hGap:hCal+hGap,:]=imgResize

        # cv2.imshow("ImageCrop",imgCrop)
        # cv2.imshow("ImageWhite",imgWhite)
        

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('s') or hands:
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg",img)
        print(counter)

    if key==ord('e'):
        break

