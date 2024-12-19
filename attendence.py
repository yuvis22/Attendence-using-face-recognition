
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Reading the images from a given folder

path = 'C:\\Users\HP\OneDrive\Desktop\\New folder\Imagesbasic'
images = []
classnames = []
mylist = os.listdir(path)

#creating a list of given images

for cl in mylist:
    currImg=cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classnames.append(os.path.splitext(cl)[0])


#Encoding the given list of images and creating a new encoded list

def find_encodings(images):
    encodedlist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodedlist.append(encode)
    return encodedlist


#Marking of attendence in a csv file

def marAttendence(name):
    with open('C:\\Users\HP\OneDrive\Desktop\\New folder\ATTENDENCE.csv','r+') as f:
        mydataList=f.readlines()
        nameList=[]
        for line in mydataList:
            entry =line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            now=datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')




EncodeListKnown=find_encodings(images)
print('Encoding completed')

#capturing and encoding the current persons image

cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    small_img=cv2.resize(img,(0,0),None,0.25,0.25)
    small_img=cv2.cvtColor(small_img,cv2.COLOR_BGR2RGB)

    facesCurrFrame=face_recognition.face_locations(small_img)
    encodeCurrFrame=face_recognition.face_encodings(small_img,facesCurrFrame)


  #matching the current encoded image to images in the provided list

    for encodeFace,faceLoc in zip(encodeCurrFrame,facesCurrFrame):
        matches =face_recognition.compare_faces(EncodeListKnown,encodeFace)
        Facedis =face_recognition.face_distance(EncodeListKnown,encodeFace)
        
        matchIndex = np.argmin(Facedis)
        
        if matches[matchIndex]:
            name = classnames[matchIndex].upper()

         #Locating the face of the person in the image and showing name of the person

            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            marAttendence(name)

    
    cv2.imshow('Webcam',img) #Showing Cam
    cv2.waitKey(1) 
