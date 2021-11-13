import cv2
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='./dataset/'
file_name=input("enter name of person:")
while True:
	ret,frame=cap.read()
	
	if ret==False:
		continue
	
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces,key=lambda f:f[2]*f[3])
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		offset=10
		face_selection=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection=cv2.resize(face_selection,(100,100))
		skip+=1
		if(skip%10==0):
			face_data.append(face_selection)
			print(len(face_data))
		
	cv2.imshow("video",frame)
	#cv2.imshow("face section",face_selection)
	key_pressed=cv2.waitKey(1) & 0XFF
	if key_pressed==ord('q'):
		break
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(dataset_path+file_name+'.npy',face_data)
print("data successfully saved at"+dataset_path+file_name+'.npy')		
cap.release()
cv2.destroyAllWindows()