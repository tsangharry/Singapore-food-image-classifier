from __future__ import division
import numpy as np
import random
from src.inference import resize_image, return_probability, return_class
from PIL import Image

def test_resize_image():
    arr = np.random.randint(low=0, high=255, size=(300, 300, 3))
    randomly_generated_image = Image.fromarray(arr.astype('uint8'))
    img = resize_image(randomly_generated_image)
    assert img.shape == (1, 299, 299, 3)


def test_return_class():
    prediction_prob_list = [0.999]
    for i in range(0,11):
        n = random.randint(1,9)/10
        prediction_prob_list.append(n)
    prediction_prob = np.array(prediction_prob_list)
    class_name = return_class(prediction_prob)
    assert class_name == 'Chilli Crab'


def test_return_probability():
    prediction_prob_list = [0.999]
    for i in range(0,11):
        n = random.randint(1,9)/10
        prediction_prob_list.append(n)
    prediction_prob = np.array(prediction_prob_list)
    probability = return_probability(prediction_prob)
    assert probability == 0.999
