import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create();
cascadePath= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");
recognizer.read('trainningData.yml')
cam = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
Id=0
while(1):
    ret, facialrec = cam.read()
    gray=cv2.cvtColor(facialrec,cv2.COLOR_BGR2GRAY)
    faces=cascadePath.detectMultiScale(gray, 1.2,5)
    if(len(faces)):
        for(x,y,w,h) in faces:
            cv2.rectangle(facialrec,(x,y),(x+w,y+h),(225,0,0),2)
            Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf<50):
                if(Id==ID_No.): 
                    Id="NAME OF THE PERSON"
            else:
                Id="unknown"
            cv2.putText(facialrec, str(Id), (x,y+h), fontface, fontscale, fontcolor)
            cv2.imshow("img",facialrec)            
    else:
        cv2.imshow("img", facialrec)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
