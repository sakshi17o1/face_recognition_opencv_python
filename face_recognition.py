import cv2
import os
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 

loading_model=cv2.face.LBPHFaceRecognizer_create()
loading_model.read("Face_recognization_system.yml")
actors_name=["Akhil","Allu Arjun","Brahmi","Nani","Prabhas","RamCharan","Sharukh" ]

cap=cv2.VideoCapture(0)

while True:
    isTrue, frame=cap.read()

    gray_img =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    face_roi = face_detector. detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3)
    for x,y,w,h in face_roi:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(168, 162, 50),3)
        crop_face=gray_img[y:y+h,x:x+w]

        label,confidence=loading_model.predict(crop_face)

        cv2.putText(frame,f"{actors_name[label]},conf:{confidence}",(x - 15, y - 15), cv2.FONT_ITALIC,1, (168, 162, 50),2)



    cv2.imshow("Face_recognization_system",frame)

    if cv2.waitKey(20) & 0xff ==ord("x"):
        break


