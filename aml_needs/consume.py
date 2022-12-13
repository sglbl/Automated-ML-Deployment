import urllib.request
import json
import os
import ssl
import base64

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
        

def encoder(image_src):
    # with open(image_src, "rb") as image_file:
    with open(image_src, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string


def encoded_to_string_to_json_file(base64_bytes):
    # third: decode these bytes to text
    # result: string (in utf-8)
    base64_string = base64_bytes.decode("utf-8")
    
    # optional: doing stuff with the data
    # result here: some dict
    raw_data = {"image": base64_string}

    # now: encoding the data to json
    # result: string
    json_data = json.dumps(raw_data, indent=2)

    # finally: writing the json string to disk
    # note the 'w' flag, no 'b' needed as we deal with text here
    with open("outputfile.json", 'w') as json_file:
        json_file.write(json_data)
    return json_data


if __name__ == "__main__":   
    # this is for using self-signed certificate in scoring service.
    allowSelfSignedHttps(True)
    base64_bytes = encoder('ss.png')

    body = "\"" + base64_bytes.decode("utf-8")  + "\"" 
    body = str.encode(body)

    # Endpoint URL
    url = 'https://ep-try3.southcentralus.inference.ml.azure.com/score'
    # AMLToken key for the endpoint
    api_key = 'API_KEY'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'dep-try13' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        print(result.decode("utf8", 'ignore'))
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
