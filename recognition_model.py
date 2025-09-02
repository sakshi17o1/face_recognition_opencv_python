import cv2
import os
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
actors_name=["Akhil","Allu Arjun","Brahmi","Nani","Prabhas","RamCharan" ]

path=r"C:\Users\pc\OneDrive\Desktop\face_recogition\images" 

labels=[]
actors_face=[]

for actors in actors_name:
    actor_folder =os.path .join (path,actors)
    actor_index=actors_name.index(actors)
    
    for images in os.listdir(actor_folder):
        actor_img_path=os.path.join(actor_folder,images)
           
        array_img=cv2.imread(actor_img_path)

        gray_img =cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)

        face_roi = face_detector. detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3)

        for x,y,w,h in face_roi:
            crop_face=gray_img[y:y+h,x:x+w]
            
            labels.append(actor_index)
            actors_face.append(crop_face)

label_array=np.array(labels)
actor_face_array=np.array(actors_face,dtype='object')

model=cv2.face.LBPHFaceRecognizer_create() 
model.train(actor_face_array,label_array)
model.save("Face_recognization_system.yml")