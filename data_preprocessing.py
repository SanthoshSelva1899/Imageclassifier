# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:48:08 2022

@author: santhosh
"""
import cv2
import os
from tensorflow.keras.preprocessing import image
face_cascade = cv2.CascadeClassifier('/Users/santh/Downloads/project1/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/santh/Downloads/project1//haarcascades/haarcascade_eye.xml')

lionel_messi="/Users/santh/Downloads/project1/Train/lionel_messi/"
maria="/Users/santh/Downloads/project1/Train/maria/"
roger="/Users/santh/Downloads/project1/Train/roger_federer/"
serena="/Users/santh/Downloads/project1/Train/serena_williams/"
virat="/Users/santh/Downloads/project1/Train/virat_kohli/"

croped="/Users/santh/Downloads/project1/cropped_train/"



def cropped_image(image_path,train,name):
    count=1
    
    for file in os.listdir(image_path):
        croppedpath=train+name+str(count) + ".png"
        path=os.path.join(image_path,file)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = img[y:y+h, x:x+w]
           eyes = eye_cascade.detectMultiScale(roi_gray)
           if len(eyes) >= 2:
               cv2.imwrite(croppedpath,roi_color)
        count=count+1

cropped_image(lionel_messi, croped,"Messi")
cropped_image(maria, croped,"maria")
cropped_image(roger, croped,"roger")
cropped_image(serena, croped,"serena")
cropped_image(virat, croped,"virat")





               
             

        