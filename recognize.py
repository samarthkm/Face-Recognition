import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = []
dir = 'Faces/Training'

for i in os.listdir(dir):
    people.append(i)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained.yml')

img = cv.imread('Faces/Testing/7.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#cv.imshow('Person',gray)

#Detection
faces_rect = haar_cascade.detectMultiScale(gray)
print(faces_rect)
for(x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+h]
    label,confidence = face_recognizer.predict(faces_roi)
    print(f'Identified as {people[label]} ,confidence: {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness = 2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected Face',img)
cv.waitKey(0)
