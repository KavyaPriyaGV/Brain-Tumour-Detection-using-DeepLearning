import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
model=load_model('Braintumor10Epochs.h5')#changing the epoch file as categorical
image=cv2.imread('C:\\Users\\Kavya\\OneDrive\\Documents\\deeplearning\\brain tumor detection\\datasets\\pred\\pred5.jpg')# if we change the img well get the [[0]] not affected
img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
#to expand the dimensions
input_img=np.expand_dims(img,axis=0)
result=model.predict(input_img)
result_final=np.argmax(result,axis=1)
print(result_final) 