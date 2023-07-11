from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
from os import listdir
from os.path import join

MODEL_NAME = 'flower.hd5'
# Our samples directory
SAMPLE_PATH = '../Lab7/sample'
# dict = {0:'persian_cats', 1:'Ragdoll', 2:'Scottish fold cats', 3:'Singapura', 4:'Sphynx cat'}
dict = {0: 'daisy', 1: 'dandelion', 2: 'roses',
        3: 'sunflowers', 4: 'tulips'}
# Takes in a loaded model, an image in numpy matrix format,
# And a label dictionary
session = tf.compat.v1.Session(graph=tf.compat.v1.Graph())


def classify(model, image):
    with session.graph.as_default():
        set_session(session)
    result = model.predict(image)
    themax = np.argmax(result)
    return dict[themax], result[0][themax], themax


# Load image
def load_image(image_fname):
    img = Image.open(image_fname)
    img = img.resize((249, 249))
    imgarray = np.array(img) / 255.0
    final = np.expand_dims(imgarray, axis=0)
    return final


# Test main
def main():
    with session.graph.as_default():
        set_session(session)
        print("Loading model from ", MODEL_NAME)
        model = load_model(MODEL_NAME)
        print("Done")

        print("Now classifying files in ", SAMPLE_PATH)
        sample_files = listdir(SAMPLE_PATH)
        for filename in sample_files:
            if filename != '.DS_Store':
                filename = join(SAMPLE_PATH, filename)
                img = load_image(filename)
                label, prob, _ = classify(model, img)
                print("We think with certainty %3.2f that image %s is %s. " % (prob, filename, label))


if __name__ == '__main__':
    main()
