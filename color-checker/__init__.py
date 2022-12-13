import logging
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import azure.functions as func


def load_model_and_predict(inp, modelFileName):
    inp = np.asarray(inp)
    model = tf.keras.models.load_model(modelFileName)
    predictions = model.predict(inp)
    red_total = 0
    green_total = 0
    blue_total = 0
    for pred in predictions:
        red_total += pred[0]*100
        green_total += pred[1]*100
        blue_total += pred[2]*100
        #print("prediction (r/g/b)%:",pred[0]*100, pred[1]*100, pred[2]*100)
    inp_x_dim, _ = inp.shape
    print("inp_x_dim:", inp_x_dim)
    red_avg = red_total/inp_x_dim
    green_avg = green_total/inp_x_dim
    blue_avg = blue_total/inp_x_dim
    return (red_avg, green_avg, blue_avg)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed  a request.')

    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    req_body = req.get_json()
    data = req_body.get('data')
    modal = req_body.get('modal')

    # print("modal:", modal)
    # print("data:", data)

    results = load_model_and_predict(data, modal)
    return func.HttpResponse(json.dumps(results), headers=headers)
