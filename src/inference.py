from __future__ import division
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def resize_image(file):
    img = file.resize((299, 299))
    img = np.divide(tf.keras.preprocessing.image.img_to_array(img),255) #convert to vector
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img, model):
    prediction_prob = model.predict(img)
    return prediction_prob


def return_class(prediction_prob):
    foods = ['Chilli Crab',
            'Curry Puff',
            'Dim Sum',
            'Ice Kacang',
            'Kaya Toast',
            'Nasi Ayam',
            'Popiah',
            'Roti Prata',
            'Sambal Stingray',
            'Satay',
            'Tau Huay',
            'Wanton Noodle']
    class_name = foods[np.argmax(prediction_prob)]
    return class_name


def return_probability(prediction_prob):
    probability = np.max(prediction_prob)
    return probability

if __name__ == '__main__':
    img_path = sys.argv[1]
    model_path = 'src/model.h5'
    model = load_model(model_path)
    img = Image.open(img_path).convert("RGB")
    img = resize_image(img)
    prediction_prob = predict_image(img, model)
    class_name = return_class(prediction_prob)
    probability = return_probability(prediction_prob)

    print({class_name, probability})

