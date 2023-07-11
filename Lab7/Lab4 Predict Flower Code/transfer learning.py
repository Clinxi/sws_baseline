import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import scipy
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import SGD
import os.path

MODEL_FILE = "flower.hd5"


def create_model(num_hidden, num_classes):
    # We get the base model using InceptionV3 and the imagenet
    # weights that was trained on tens of thousands of images.
    base_model = InceptionV3(include_top=False, weights='imagenet')
    # Get the output layer, then does an average pooling of this
    # output, and feeds it to a final Dense layer that we
    # will train
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_hidden, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # Set base_model layers to be non_trainable, so we focus
    # our training only in the Dense layer. This lets us
    # adapt much faster and doesn't corrupt the weights that
    # were already trained on imagenet.
    for layer in base_model.layers:
        layer.trainable = False
    # Create a Functional Model (as opposed to the usual
    # Sequential Model that we create
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Loads an existing model file, then sets only the last
# 3 layers (which we added) to trainable.
def load_existing(model_file):
    # Load the model
    model = load_model(model_file)
    # Set only last 3 layers as trainable
    numlayers = len(model.layers)
    for layer in model.layers[:numlayers - 3]:
        layer.trainable = False
    # Set remaining layers to be trainable.
    for layer in model.layers[numlayers - 3:]:
        layer.trainable = True
    return model


# Trains a model. Creats a new model if it doesn't already exist.
def train(model_file, train_path, validation_path, num_hidden=200, num_classes=5, steps=32, num_epochs=20):
    # If an existing model exists, we load it. Otherwise we create a
    # new model from scratch
    if os.path.exists(model_file):
        print("\n*** Existing model found at %s. Loading***\n\n" % model_file)
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model(num_hidden, num_classes)
    # Since we have multiple categories and a softmax output
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # Create a checkpoint to save the model after every epoch
    checkpoint = ModelCheckpoint(model_file)
    # Now we create a generator. This will take our image data, rescale it,
    # shear it, zoom in and out to create additional images for training.
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    # Image generator for test data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Now we tell the generator where to get the images from.
    # We also scale the images to 249x249 pixels.
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(249, 249), batch_size=5, class_mode=
        "categorical")
    # We do the same for the validation set .
    validation_generator = test_datagen.flow_from_directory(
        validation_path, target_size=(249, 249), batch_size=5,
        class_mode='categorical')
    # Finally we train the neural network
    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=50)
    # Train last two layers
    # Now we twek the training by freezing almost all the layers and
    # just train the topmost layer
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    model.compile(optimizer=SGD(learning_rate=0.00001, momentum=0.9), loss='categorical_crossentropy')
    model.fit(train_generator, steps_per_epoch=steps,
              epochs=num_epochs,
              callbacks=[checkpoint],
              validation_data=validation_generator,
              validation_steps=50)


def main():
    train(MODEL_FILE, train_path="/Users/user/Downloads/SWS3009/Cats", validation_path="/Users/user/Downloads/SWS3009/Cats", steps=120, num_epochs=10)


if __name__ == '__main__':
    main()
