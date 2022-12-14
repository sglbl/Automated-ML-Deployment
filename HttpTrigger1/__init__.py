import io
import os
import json
import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image
import base64
import logging
import azure.functions as func


class InferenceModel():
    def __init__(self, model_path):
        # Load the model from the path
        self.model = tf.keras.models.load_model(model_path)
 
 
    def face_border_detector(self, image):
        # get xml from repo
        face_cascade = cv.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        biggest_face = [0,0,0,0]
        for i, (x, y, w, h) in enumerate(faces):
            if (biggest_face[2] - biggest_face[0]) < (w - x):
                biggest_face = [x, y, w, h]
                
        [x,y,w,h] = biggest_face
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv.imshow("image", image)
        # cv.waitKey(0)
        return biggest_face


    # Decode image from base64 to pillow and opencv images
    def preprocess_image(self, image_encoded):     
        pil_image = Image.open(io.BytesIO(base64.b64decode(image_encoded)))
        pil_image_gray = pil_image.resize((48,48)).convert('L')
        # pil_image_gray.show()
        image_np = (255 - np.array(pil_image_gray.getdata())) / 255.0
        cv_image = np.array(pil_image) # converting PIL image to cv2 image
        cv_image = cv_image[:, :, ::-1].copy()  # convert from RGB to BGR

        return cv_image, image_np.reshape(-1,48,48,1)
 
 
    def predict(self, req_body):
        opencv_image, image_data = self.preprocess_image(req_body)
        prediction = self.model.predict(image_data)
        # print(prediction.tolist()[0][0], "Predict list")
        predicted_age = int(prediction.tolist()[0][0])
        face_borders_and_age = self.face_border_detector( opencv_image )
        face_borders_and_age.append(predicted_age)
        return face_borders_and_age


def int32_to_int(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed  a request.')

    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    # req_body = req.get_json()
    req_body = req.get_data(False)
    image = req_body.get('image')
    model_path = req_body.get('model_path')

    print("modal:", model)
    print("image:", image)

    # Model path will be got by flutter function
    global model
    model = InferenceModel(model_path)        
    preds = model.predict(image)
    # preds = model.predict(req_body)
    results = json.dumps({"preds": preds}, default=int32_to_int)
    
    return func.HttpResponse(json.dumps(results), headers=headers)

