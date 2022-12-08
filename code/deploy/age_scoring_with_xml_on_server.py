import os
import json
import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image
import requests
#from inference_model import InferenceModel
 
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
 

class InferenceModel():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
 
    def face_border_detector(self, image):
        # download xml from server
        link = "https://www.studenti.famnit.upr.si/~76210123/76210123/xml/haarcascade_frontalface_default2.xml"
        r = requests.get(link, allow_redirects=True)
        open('haarcascade_frontalface_default2.xml', 'wb').write(r.content)
        # end of download      
        
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default2.xml')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        biggest_face = [0,0,0,0]
        for i, (x, y, w, h) in enumerate(faces):
            if (biggest_face[2] - biggest_face[0]) < (w - x):
                biggest_face = [x, y, w, h]
                
        [x,y,w,h] = biggest_face
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return biggest_face
 
 
    def _preprocess_image(self, image_bytes):
        image = Image.open(image_bytes)        
        image = image.resize((48,48)).convert('L')
        # image.show()
        image_np = (255 - np.array(image.getdata())) / 255.0
 
        return image_np.reshape(-1,48,48,1)
 
    def predict(self, image_bytes):
        image_data = self._preprocess_image(image_bytes)
        prediction = self.model.predict(image_data)
        
        face_borders = self.face_border_detector(cv.imread(image_bytes))
        # print("Prediction: ", prediction)
        # print("Face borders: ", face_borders)
        face_borders.append(prediction)
        # return [prediction, face_borders]
        return face_borders


def init():
    global model
    model_name = "age_model_a"
    model_path = model_name
    model = InferenceModel(model_path)

@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse(f"Unsupported verb: {request.method}", 400)
 
    image_data = request.files['image']
    preds = model.predict(image_data)
    
    return AMLResponse(json.dumps({"preds": preds.tolist()}), 200)
    