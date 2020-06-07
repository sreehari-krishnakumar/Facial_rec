import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path='dataSet'

def getImagesWithID(path):
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
	faces=[]
	IDs=[]
	for imagePath in imagePaths:
		faceImg=Image.open(imagePath).convert('L');
		faceNp=np.array(faceImg,'uint8')
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		print (ID)
		IDs.append(ID)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)
		print (ID)

	return np.array(IDs),faces

Ids,faces=getImagesWithID(path)
recognizer.train(faces,Ids)
recognizer.save('trainningData.yml')
cv2.destroyAllWindows()
