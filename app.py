from __future__ import division, print_function

# coding=utf-8
import os

import cv2 as cv
import numpy as np
import tensorflow as tf
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from models.models.model import SuperResolutionNet
import math

# Define a flask app
app = Flask(__name__)

# tf session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# Load trained model
net = SuperResolutionNet((128, 128, 3), (128, 128, 3), 0.05, 0.7)
total_count = 0
model_path = './models/tmp/model.ckpt'
saver = tf.train.Saver()
saver.restore(sess, model_path)
print('Model loaded. Start serving...')


def parse_img(path, num=4, dim=100):
    img = cv.imread(path)
    img = img / 255.0

    images = []
    h_dim = math.ceil(img.shape[0] / dim)
    w_dim = math.ceil(img.shape[1] / dim)
    img_h, img_w, img_d = img.shape
    for height in range(h_dim):
        for width in range(w_dim):
            start_x = height * (dim - 14)
            if start_x > img_h - 128:
                start_x = img_h - 128
            start_y = width * (dim - 14)
            if start_y > img_w - 128:
                start_y = img_w - 128
            end_x = start_x + 128
            end_y = start_y + 128
            temp_img = img[start_x: end_x, start_y: end_y, :]
            temp_img = temp_img.reshape(-1, 128, 128, 3)
            # print('height', height, 'width', width, 'shape', temp_img.shape, 'start_x', start_x, 'end_x', end_x, 'start_y', start_y, 'end_y', end_y)
            images.append(temp_img)
    return images


def convert_paths_to_nd_array(paths, num):
    images = []
    for path in paths:
        img = parse_img(path, num, 100)
        images.extend(img)
    images = np.concatenate(images)
    return images


def plot_img(images, count, dim):
    global total_count
    total_count = total_count + 1
    # padding = 14
    image = np.zeros([128 * 4, 128 * 4, 3])
    temp_h = 0
    for height in range(dim):
        temp_w = 0
        for width in range(dim):
            if height == 0:
                start_x = 0
                end_x = start_x + 100
            elif height == (dim - 1):
                start_x = 128 - 68
                end_x = 128
            else:
                start_x = 14
                end_x = start_x + 100
            if width == 0:
                start_y = 0
                end_y = start_y + 100
            elif width == (dim - 1):
                start_y = 128 - 68
                end_y = 128
            else:
                start_y = 14
                end_y = start_y + 100

            temp = images[dim * height + width][start_x: end_x, start_y: end_y, :]
            # print(temp.shape, start_x, end_x, start_y, end_y)
            h, w, d = temp.shape
            image[temp_h: temp_h + h, temp_w: temp_w + w, :] = temp
            if width == 0:
                temp_w = temp_w + 100
            else:
                temp_w = temp_w + 100 - 14
        if height == 0:
            temp_h = temp_h + 100
        else:
            temp_h = temp_h + 100 - 14

    image_decode = image * 255
    cv.imwrite('./static/images/' + str(count) + '.jpg', image_decode)
    return str(count) + '.jpg'


def model_predict(img_path, model):
    # describe = []
    paths = [img_path]
    images = convert_paths_to_nd_array(paths, 4)
    img_dict = {model.inputs: images}
    decode_image = sess.run(model.decode, feed_dict=img_dict)
    return plot_img(decode_image, total_count, 6)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, net)
        # result = str(preds[0] + '\n' + preds[1])
        return preds
    return None


if __name__ == '__main__':
    app.run(port=5088, debug=True)

