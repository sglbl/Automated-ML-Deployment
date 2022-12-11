import io
import os
import json
import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image
import requests
import base64
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
 
# from django.core.files.base import ContentFile

class InferenceModel():
    def __init__(self, model_path):
        # Load the model from the path
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


def init():
    global model
    model_name = "age_model_a"
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name)
    model = InferenceModel(model_path)


@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse(f"Unsupported verb: {request.method}", 400)
  
    # exc_occured = False
    # try:
    #     req_body = request.get_data(False)    
    #     preds = model.predict(req_body)
    # except Exception as e:
    #     print("Exception occured in get_data(): ", e)
    #     exc_occured = True        
    
    req_body = request.get_data(False)    
    # preds = model.predict(req_body["image"])
    print(req_body[:25])
    preds = model.predict(req_body)

    dumped_preds = json.dumps({"preds": preds}, default=int32_to_int)
    return AMLResponse(dumped_preds, 200)
    

# def encoder(self, image_src):
#     # with open(image_src, "rb") as image_file:
#     with open(image_src, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     return encoded_string


# def init_local():
#     global model
#     model_path = "../age_model_a"
#     model = InferenceModel(model_path)
    
#     # encoded_photo = model.encoder("dede.jpg")
#     with open('outputfile.json') as json_file:
#         json_ob = json.load(json_file)  # read outputfile as json
#     encoded_photo = json_ob["image"]
    
#     preds = model.predict(encoded_photo)
#     dumped_preds = json.dumps({"preds": preds}, default=int32_to_int)
#     print("Dumped preds: ", dumped_preds)    
    
    
# if __name__ == "__main__":
#     init_local()