import cv2,os,time
import numpy as np
import subprocess
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create();
cascadePath= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");
recognizer.read('trainningData.yml')
cam = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
i=1
Id=0
while(1):
    ret, facialrec = cam.read()
    gray=cv2.cvtColor(facialrec,cv2.COLOR_BGR2GRAY)
    faces=cascadePath.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(facialrec,(x,y),(x+w,y+h),(225,0,0),2)
        Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        print ("id",Id)
        if(conf<50):
            if(Id==2):
                Id="Username"
        elif(conf>50):
            Id="Unknown"
        cv2.putText(facialrec, str(Id), (x,y+h), fontface, fontscale, fontcolor)
        cv2.imshow("img",facialrec)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
# and release the output
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
