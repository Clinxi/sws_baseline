import paho.mqtt.client as mqtt
import json
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
from os import listdir
from os.path import join

classes = ["daisy", "dandelion", "poses", "sunflowers", "tulips"]
MODEL_NAME = 'flower.hd5'
session = tf.compat.v1.Session(graph=tf.compat.v1.Graph())

def loading_model():
    with session.graph.as_default():
        set_session(session)
        model = load_model(MODEL_NAME)
        return model


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Group_9/IMAGE/classify")
    else:
        print("Connection failed with code: %d." % rc)


def classify_flower(filename, data):
    print("Start classifying")
    # with session.graph.as_default():
    #     set_session(session)
    result = model.predict(data)
    themax = np.argmax(result)
    print("Done.")
    return {"filename": filename, "prediction": classes[themax], "score": result[0][themax], "index": themax}


def on_message(client, userdata, msg):
    # Payload is in msg. We convert it back to a Python dictionary
    recv_dict = json.loads(msg.payload)
    # Recreate the data
    img_data = np.array(recv_dict["data"])
    print(img_data)
    result = classify_flower(recv_dict["filename"], img_data)
    print("Sending results: ", result)
    client.publish("Group_9/IMAGE/predict", str(result)) # json.dumps(result))


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client


def main():
    # with session.graph.as_default():
    #     set_session(session)
    print("Loading model from", MODEL_NAME)
    global model
    model = load_model(MODEL_NAME)
    print("Done")
    setup("localhost")
    while True:
        pass

  
if __name__ == '__main__':
    main()
