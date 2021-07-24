import os
import cv2 as cv
import numpy as np

people = []
dir = 'Faces/Training'
haar_cascade = cv.CascadeClassifier('haar_face.xml')
for i in os.listdir(dir):
    people.append(i)


features = []
identifier = []

def create_train():
   for person in people:
       path =  os.path.join(dir,person)
       label = people.index(person)
       
       for img in os.listdir(path):
           img_path = os.path.join(path,img)
           img_array = cv.imread(img_path)
           gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
           faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
           for(x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                identifier.append(label)

create_train()

features = np.array(features,dtype='object')
identifier = np.array(identifier)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,identifier)

face_recognizer.save('trained.yml')
np.save('features.npy',features)
np.save('identifier.npy',identifier)