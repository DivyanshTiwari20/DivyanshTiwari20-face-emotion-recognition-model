import os
import tensorflow as tf
import keras
from keras.src.utils import load_img
from time import sleep
from keras_preprocessing.image import img_to_array
from keras_preprocessing import image
import cv2
import numpy as np
import h5py

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path for the XML and model files
haar_cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
model_path = os.path.join(current_dir, 'model.h5')

# Use these paths to load the classifier and model
face_classifier = cv2.CascadeClassifier(haar_cascade_path)
classifier = keras.models.load_model(model_path)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

# Create a named window
cv2.namedWindow('Emotion Detector')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.imshow('Emotion Detector',frame)
    
    # Check for 'q' key press or if window was closed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Emotion Detector', cv2.WND_PROP_VISIBLE) < 1:
        break

print("Releasing camera and closing windows...")
cap.release()
cv2.destroyAllWindows()

# Ensure all windows are closed
for i in range(1,5):
    cv2.waitKey(1)

print("Program has exited.")