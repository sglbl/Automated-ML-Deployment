import io
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
        face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    
        # cv.imwrite("gray.jpg", image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        biggest_face = [0,0,0,0]
        if len(faces) == 0:
            logging.info("No faces found in file1")
            face_cascade2 = cv.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
            faces = face_cascade2.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                #print("No faces found in file2")
                face_cascade3 = cv.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
                faces = face_cascade3.detectMultiScale(gray, 1.1, 4)
                logging.info(faces)
                if len(faces) == 0:
                    logging.info("No faces found in file3")
                    face_cascade4 = cv.CascadeClassifier('models/haarcascade_frontalface_alt_tree.xml')
                    faces = face_cascade4.detectMultiScale(gray, 1.1, 4)

        logging.info(f"Faces : {faces}")
        for i, (x, y, w, h) in enumerate(faces):
            logging.info("iter")
            if abs(biggest_face[2]) < abs(w):
                biggest_face = [x, y, w, h]
                logging.info(f"Biggest face : {biggest_face}")
        
        logging.info(f"At the end {biggest_face}")
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
    logging.info('Python HTTP trigger function processed a request.')
    
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }
        
    model_path = req.params.get('model_path')
    image = req.params.get('image')
    if not model_path:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            model_path = req_body.get('model_path')
            image = req_body.get('image')    

    # Model path will be get by flutter function
    global model
    model = InferenceModel(model_path)        
    preds = model.predict(image)
    # preds = model.predict(req_body)
    results = json.dumps({"preds": preds}, default=int32_to_int)
    
    return func.HttpResponse(json.dumps(results), headers=headers)

