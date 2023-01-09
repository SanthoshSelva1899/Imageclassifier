# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:04:58 2022

@author: santhosh
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications import imagenet_utils

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import base64
import io

face_cascade = cv2.CascadeClassifier('/Users/santh/Downloads/project1/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/santh/Downloads/project1//haarcascades/haarcascade_eye.xml')
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='/Users/santh/Downloads/project1/finally_model_santhosh.h5'
resized_folder="/Users/santh/Downloads/project1/resized_folder/"
model = tf.keras.models.load_model(
       (MODEL_PATH),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

cropped_uploads='/Users/santh/Downloads/project1/static/'

def cropped_image(image_path,name):
   img = cv2.imread(image_path)
   croppedpath=os.path.join(cropped_uploads,name)
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.3, 5)
   for (x,y,w,h) in faces:
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = img[y:y+h, x:x+w]
       eyes = eye_cascade.detectMultiScale(roi_gray)
       cv2.imwrite(croppedpath,roi_color)
       if len(eyes) >= 2:
           cv2.imwrite(croppedpath,roi_color)
   return croppedpath


def model_predict(img_path, model,file):
    input_image = cv2.imread(img_path)

    input_image_resize = cv2.resize(input_image, (224,224))

    input_image_scaled = input_image_resize/255

    image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

    predis = model.predict(image_reshaped)
    
    return predis


@app.route('/', methods=['GET','POST'])
def index():
    # Main page
    return render_template('index_2.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        d=f.filename

        # Make prediction
        pathprocessed = cropped_image(file_path,d)
        pred=model_predict(pathprocessed, model,d)
        print(pred)
        preds=pred.tolist()
        pome=preds[0][0]
        porog=preds[0][1]
        pom=preds[0][2]
        pos=preds[0][3]
        pov=preds[0][4]
        img=Image.open(pathprocessed)
        data=io.BytesIO()
        img.save(data,"JPEG")
        encode_img_data=base64.b64encode(data.getvalue())
        
        return render_template('base_2.html',messi=pome,roger=porog,maria=pom,serena=pos,virat=pov,user_image=encode_img_data.decode("UTF-8"))
        


if __name__ == '__main__':
    app.run(debug=True)
