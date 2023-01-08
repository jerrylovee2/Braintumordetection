import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread('E:\\BrainTumor Classification DL\\BrainTumor Classification DL\\uploads\\pred0.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict_step(input_img)
print(result)




