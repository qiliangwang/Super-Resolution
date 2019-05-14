import os
import random

import cv2 as cv
import numpy as np
import skimage
import skimage.io
import skimage.transform


def prepare_image(folder_dir='D:\image'):
    image_list = []
    find_image(folder_dir, image_list)
    return image_list


def find_image(file_dir, file_list):
    results = [os.path.join(file_dir, result) for result in os.listdir(file_dir)]
    for result in results:
        if os.path.isdir(result):
            find_image(result, file_list)
        else:
            file_list.append(result)
    pass


def save_image(image, image_dir, image_number):
    base_dir = '../images'
    (file_path, complete_filename) = os.path.split(image_dir)
    (filename, extension) = os.path.splitext(complete_filename)
    new_filename = str(image_number) + str(extension)
    original_512 = os.path.join(base_dir, 'original', new_filename)
    print('save original_512' + original_512)
    image_resize = resize_image(image, original_512, edge=512)

    small_512 = os.path.join(base_dir, 'small512', new_filename)
    print('save small_512' + small_512)
    resize_data(image_resize, small_512)

    small_128 = os.path.join(base_dir, 'small256', new_filename)
    print('save small_128' + small_128)
    resize_data_2(image_resize, small_128)
    pass


def resize_data(image,  image_dir):
    image_resize = skimage.transform.resize(image, (256, 256), mode='constant')
    new_image_resize = skimage.transform.resize(image_resize, (512, 512), mode='constant')
    skimage.io.imsave(image_dir, new_image_resize)
    pass


def resize_data_2(image,  image_dir):
    image_resize = skimage.transform.resize(image, (256, 256), mode='constant')
    skimage.io.imsave(image_dir, image_resize)
    pass


def resize_image(image,  image_dir, edge=512):
    short_edge = min(image.shape[:2])
    start_x = int((image.shape[1] - short_edge) / 2)
    start_y = int((image.shape[0] - short_edge) / 2)
    crop_image = image[start_y: start_y + short_edge, start_x: start_x + short_edge]
    image_resize = skimage.transform.resize(crop_image, (edge, edge), mode='constant')
    skimage.io.imsave(image_dir, image_resize)
    return image_resize
    pass


def convert_image(info):
    # image_number = 0
    number, image_dir = info.split("|||")
    try:
        image = skimage.io.imread(image_dir)
        save_image(image, image_dir, number)
    except:
        print(image_dir + " not an image")
    # for image_dir in image_list:

    pass


def main():
    # image_list = prepare_image()
    # new_images = []
    # for index in range(len(image_list)):
    #     new_images.append(str(index) + "|||" + image_list[index])
    # pool = Pool(16)
    # pool.map(convert_image, new_images)
    # convert_image(new_images[0])
    gen_mask('../images/original/0.jpg')
    pass


def gen_mask(image_dir):
    img = cv.imread(image_dir)
    # random mask
    for num in range(50):
        rand_x = random.randint(0, img.shape[0] - 10)
        rand_y = random.randint(0, img.shape[1] - 10)
        img[rand_x:rand_x + 10, rand_y: rand_y + 10] = np.zeros((10, 10, 3), np.uint8)
    while True:
        cv.imshow('image', img)
        if cv.waitKey(10) & 0xFF == 27:  # ‘esc’退出
            break


if __name__ == '__main__':
    main()
