import os
from glob import glob
from random import shuffle

import numpy as np
import skimage.io
import skimage.transform


def load_image(path):
    try:
        img = skimage.io.imread(path).astype(float)
        # TODO http://scikit-image.org/docs/dev/api/skimage.color.html rgb2gray
        # TODO cropping.
        img = skimage.transform.resize(img, (224, 224), mode='constant')
    except:
        return None
    if img is None:
        return None
    if len(img.shape) < 2:
        return None
    if len(img.shape) == 4:
        return None
    if len(img.shape) == 2:
        img = np.tile(img[:, :, None], 3)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] > 4:
        return None

    img /= 255.
    return img


def next_batch(x_y, index, batch_size):
    has_reset = False
    index *= batch_size
    updated_index = index % len(x_y)
    if updated_index + batch_size > len(x_y):
        updated_index = 0
        has_reset = True
    beg = updated_index
    end = updated_index + batch_size
    output = x_y[beg:end]
    x = np.array([e[0] for e in output])
    y = np.array([e[1] for e in output])
    return x, y, has_reset


def read_dataset(folder, max_num_training_images, max_num_testing_images, class_mapper):
    training_inputs = read_set(folder, 'train', max_num_training_images, class_mapper)
    testing_inputs = read_set(folder, 'test', max_num_testing_images, class_mapper)
    return training_inputs, testing_inputs


def read_set(folder, phase, max_num_of_images, class_mapper):
    images_folder = os.path.join(folder, phase)
    inputs = []
    list_images = glob(images_folder + '/**/*.png')
    shuffle(list_images)
    for i, image_name in enumerate(list_images):
        if len(inputs) >= max_num_of_images:
            break
        class_name = image_name.split('/')[-2]
        if i % 100 == 0:
            print(i)
        inputs.append([load_image(image_name), class_mapper[class_name]])  # TODO make them 256x256
    return inputs


def compute_mean_not_optimised(inputs):
    matrix_all_images = []
    for image, label in inputs:
        matrix_all_images.append(image)
    return np.mean(np.array(matrix_all_images), axis=0)


def compute_mean(inputs):
    image_mean = np.array(inputs[0][0])
    image_mean.fill(0)
    for i, (image, label) in enumerate(inputs):
        image_mean += image
        if i % 100 == 0:
            print(i)
    return image_mean / len(inputs)


def subtract_mean(inputs, mean_image):
    new_inputs = []
    for image, label in inputs:
        new_inputs.append([image - mean_image, label])
    return new_inputs
