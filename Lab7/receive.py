import paho.mqtt.client as mqtt
import numpy as np
import json

classes = ["daisy", "dandelion", "poses", "sunflowers", "tulips"]


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Group_99/IMAGE/classify")
    else:
        print("Connection failed with code: %d." % rc)


def classify_flower(filename, data):
    print("Start classifying")
    win = 4
    print("Done.")
    return {"filename": filename, "prediction": classes[win], "score": 0.99, "index": win}


def on_message(client, userdata, msg):
    # Payload is in msg. We convert it back to a Python dictionary
    recv_dict = json.loads(msg.payload)
    # Recreate the data
    img_data = np.array(recv_dict["data"])
    result = classify_flower(recv_dict["filename"], img_data)
    print("Sending results: ", result)
    client.publish("Group_99/IMAGE/predict", json.dumps(result))


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client


def main():
    setup("127.0.0.1")
    while True:
        pass


if __name__ == '__main__':
    main()
