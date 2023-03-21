from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def gen_frames():
    # cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)  #Setting the maximum hands

    prediction =  None
    classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
    labels = ["1", "2" , "3", "4", "5", "6", "7" , "8" , "9", "A", "B" , "C" , "D", "E", "F", "G", "H" , "I", 'J','K',"L", "M" , "N","O", "P", "Q", "R", "S","T","U", "V","W","X","Y","Z" ]
    offset = 20
    imgSize = 350
    counter =0


    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
            # Change The Slicing According The Requirement
            imgCrop = img[y-offset-10: y+h+offset+120,x-offset-10:x+w+offset+250]# imgCrop = img[10: y+h+offset+100,10:x+w+offset+200]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w

            prediction,index = classifier.getPrediction(img)
            # print(prediction,index)
            cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)
            cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),2)
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)  


        elif len(hands)==2:
            off = 20
            hand = hands[1]
            x,y,w,h = hand['bbox'] #Bounding Box information

            imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
            # Change The Slicing According The Requirement
            imgCrop = img[10: y+h+off+120,10:x+w+off+250]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w

            prediction,index = classifier.getPrediction(img)
            # print(prediction,index)
            cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)
            cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),2)
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        imgOutput = buffer.tobytes()

            # yield the frame in byte format
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')

    # while True:
    #     success, frame = camera.read()  # read the camera frame
    #     if not success:
    #         break
    #     else:
    #         # convert the frame to byte format
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()

    #         # yield the frame in byte format
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
