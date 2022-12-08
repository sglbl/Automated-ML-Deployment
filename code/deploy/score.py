import os
import joblib
import numpy as np
import argparse

from sklearn.svm import SVC
from azureml.core import Model
from azureml.monitoring import ModelDataCollector
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType


# # The init() method is called once, when the web service starts up.
# # Typically you would deserialize the model file, as shown here using joblib,
# # and store it in a global variable so your run() method can access it later.
# def init():
#     global model
#     global inputs_dc, prediction_dc
#     # The AZUREML_MODEL_DIR environment variable indicates
#     # a directory containing the model file you registered.
#     model_file_name = "model.pkl"
#     model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR"), model_file_name)
#     model = joblib.load(model_path)
#     inputs_dc = ModelDataCollector("sample-model", designation="inputs", feature_names=["feat1", "feat2", "feat3", "feat4"])
#     prediction_dc = ModelDataCollector("sample-model", designation="predictions", feature_names=["prediction"])


# # The run() method is called each time a request is made to the scoring API.
# # Shown here are the optional input_schema and output_schema decorators
# # from the inference-schema pip package. Using these decorators on your
# # run() method parses and validates the incoming payload against
# # the example input you provide here. This will also generate a Swagger
# # API document for your web service.
# @input_schema('data', NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4]])))
# @output_schema(StandardPythonParameterType({'predict': [['Iris-virginica']]}))
# def run(data):
#     # Use the model object loaded by init().
#     result = model.predict(data)
#     inputs_dc.collect(data) #this call is saving our input data into Azure Blob
#     prediction_dc.collect(result) #this call is saving our input data into Azure Blob

#     # You can return any JSON-serializable object.
#     return { "predict": result.tolist() }


################################################



# # The init() method is called once, when the web service starts up.
# # Typically you would deserialize the model file, as shown here using joblib,
# # and store it in a global variable so your run() method can access it later.
# def init():
#     global model
#     global inputs_dc, prediction_dc
#     # The AZUREML_MODEL_DIR environment variable indicates
#     # a directory containing the model file you registered.
#     # model_file_name = "saved_model.pb"
#     # model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR"), model_file_name)
#     model_path = "../../models/age_model_a/saved_model.pb"
#     model = joblib.load(model_path)
#     inputs_dc = ModelDataCollector("age_model_a", designation="inputs", feature_names=["feat1", "feat2", "feat3", "feat4"])
#     prediction_dc = ModelDataCollector("age_model_a", designation="predictions", feature_names=["prediction"])


# # The run() method is called each time a request is made to the scoring API.
# # Shown here are the optional input_schema and output_schema decorators
# # from the inference-schema pip package. Using these decorators on your
# # run() method parses and validates the incoming payload against
# # the example input you provide here. This will also generate a Swagger
# # API document for your web service.
# @input_schema('data', NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4]])))
# @output_schema(StandardPythonParameterType({'predict': [['Iris-virginica']]}))
# def run(data):
#     # Use the model object loaded by init().
#     result = model.predict(data)
#     inputs_dc.collect(data) #this call is saving our input data into Azure Blob
#     prediction_dc.collect(result) #this call is saving our input data into Azure Blob

#     # You can return any JSON-serializable object.
#     return { "predict": result.tolist() }


#############################################

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



# The init() method is called once, when the web service starts up.
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model
    global inputs_dc, prediction_dc
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    # model_file_name = "saved_model.pb"
    # model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR"), model_file_name)
    model_path = "../../models/age_model_a/saved_model.pb"
    # model = joblib.load(model_path)
    model = InferenceModel(model_path)


# The run() method is called each time a request is made to the scoring API.
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.
# @input_schema('data', NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4]])))
# @output_schema(StandardPythonParameterType({'predict': [['Iris-virginica']]}))
# def run(data):
#     # Use the model object loaded by init().
#     result = model.predict(data)
#     inputs_dc.collect(data) #this call is saving our input data into Azure Blob
#     prediction_dc.collect(result) #this call is saving our input data into Azure Blob

#     # You can return any JSON-serializable object.
#     return { "predict": result.tolist() }
@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse(f"Unsupported verb: {request.method}", 400)
 
    image_data = request.files['image']
    preds = model.predict(image_data)
    
    return AMLResponse(json.dumps({"preds": preds.tolist()}), 200)
    