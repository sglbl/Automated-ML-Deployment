import base64
import io
import cv2
import sys
import os
import json
import numpy as np
from PIL import Image
import base64
import logging
import azure.functions as func
from keras.utils.data_utils import get_file

sys.path.insert(0, 'HttpTrigger2')
from wide_resnet import WideResNet

class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    WRN_WEIGHTS_PATH = "models/age_model_c/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "models/age_model_c").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    # Decode image from base64 to pillow and opencv images
    def preprocess_image(self, image_encoded):     
        pil_image = Image.open(io.BytesIO(base64.b64decode(image_encoded)))
        pil_image_gray = pil_image.resize((64,64))
        # pil_image_gray.show()
        image_np = (255 - np.array(pil_image_gray.getdata())) / 255.0
        cv_image = np.array(pil_image) # converting PIL image to cv2 image
        cv_image = cv_image[:, :, ::-1].copy()  # convert from RGB to BGR

        return cv_image, image_np.reshape(-1,64,64,1)
    
    def encoder(self, image_src):
        # with open(image_src, "rb") as image_file:
        with open(image_src, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string

    def face_border_detector(self, image):
        # get xml from repo
        face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    
        cv2.imwrite("gray.jpg", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        biggest_face = [0,0,0,0]
        if len(faces) == 0:
            logging.info("No faces found in file1")
            face_cascade2 = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
            faces = face_cascade2.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                #print("No faces found in file2")
                face_cascade3 = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
                faces = face_cascade3.detectMultiScale(gray, 1.1, 4)
                logging.info(faces)
                if len(faces) == 0:
                    logging.info("No faces found in file3")
                    face_cascade4 = cv2.CascadeClassifier('models/haarcascade_frontalface_alt_tree.xml')
                    faces = face_cascade4.detectMultiScale(gray, 1.1, 4)

        logging.info(f"Faces : {faces}")
        for i, (x, y, w, h) in enumerate(faces):
            if abs(biggest_face[2]) < abs(w):
                biggest_face = [x, y, w, h]
                logging.info(f"Biggest face : {biggest_face}")
        
        logging.info(f"At the end {biggest_face}")
        [x,y,w,h] = biggest_face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # # Show image with face border
        
        return biggest_face

    def predictor(self, req_body):
        opencv_image, image_data = self.preprocess_image(req_body)
        resized_img = cv2.resize(opencv_image, (64, 64), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        face_imgs = np.empty((1, 64, 64, 3))
        face_imgs[0,:,:,:] = resized_img
        prediction = self.model.predict(face_imgs)
        
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_age = int(prediction[1].dot(ages).flatten()[0])
        # print("Predict result:", prediction[1].dot(ages).flatten())
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
    # image_path = "C:\Apparatus\GTU\Year4\CSE495\Project\models\ex_images\\20_girl.jpeg"
    # encoded_photo = model.encoder(image_path)
    # model = InferenceModel(model_path)        
    model = FaceCV(16,8)
    
    preds = model.predictor(image)
    # preds = model.predict(req_body)
    results = json.dumps({"preds": preds}, default=int32_to_int)
    
    return func.HttpResponse(json.dumps(results), headers=headers)

