from tensorflow.keras.layers import Input,Dense,Lambda,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import tensorflow_hub as hub
import os
os.chdir("/Users/santh/Downloads/project1/")

image_size=[224,224]
train_data="/Users/santh/Downloads/project1/cropped_train/"
test_data="/Users/santh/Downloads/project1/cropped_test/"
resized_folder="/Users/santh/Downloads/project1/resized_folder/"
input_image_path="/Users/santh/Downloads/project1/cropped_uploads/sharapova-hits-the-practice-courts-and-met-ball-kids.jpg"

label=[]

for file in os.listdir(train_data):
    name= file
    if name[0:3]=="Mes":
        label.append(0)
    elif name[0:3]=="rog":
        label.append(1)
    elif name[0:3]=="mar":
        label.append(2)
    elif name[0:3]=="ser":
        label.append(3)
    else:
        label.append(4)
        
y=np.asarray(label) 

print(y)


 
for file in os.listdir(train_data):
    image_path=train_data+file
    img=Image.open(image_path)
    img=img.resize((224,224))
    newpath=resized_folder+file
    img.save(newpath)
    
train_array=np.array([cv2.imread(resized_folder+file) for file in os.listdir(resized_folder)])





x_train,x_test,y_train,y_test=train_test_split(train_array,y,test_size=0.2,random_state=2)

x_train_scaled=x_train/255
x_test_scaled=x_test/255

mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/3'

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)

num_of_classes = 5

model = tf.keras.Sequential([
    
    pretrained_model,
    tf.keras.layers.Dense(num_of_classes)

])


model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer='adam',metrics=['accuracy'])


r=model.fit(x_train_scaled,y_train,epochs=20)

from tensorflow.keras.models import load_model

model.save('finally_model_santhosh.h5')

input_image = cv2.imread(input_image_path)

input_image_resize = cv2.resize(input_image, (224,224))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

input_prediction = model.predict(image_reshaped)
print(type(input_prediction))
print(input_prediction)
pred=np.argmax(input_prediction)

print(pred)
