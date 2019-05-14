import os
import cv2 as cv
import numpy as np


class SuperResolutionData:
    def __init__(self, base_dir='../images'):

        self.data_dir = os.path.join(base_dir, 'small512')
        self.label_dir = os.path.join(base_dir, 'original')

        self.data = [os.path.join(self.data_dir, img_dir) for img_dir in os.listdir(self.data_dir)]
        self.label = [os.path.join(self.label_dir, img_dir) for img_dir in os.listdir(self.label_dir)]

    @staticmethod
    def parse_img(path, num, dim):
        img = cv.imread(path)
        img = img / 255.0

        images = []
        len_x = img.shape[0] // num
        for height in range(num):
            for width in range(num):
                temp_img = img[height * len_x:height * len_x + len_x, width * len_x:width * len_x + len_x, :]
                temp_img = temp_img.reshape(-1, len_x, len_x, 3)
                images.append(temp_img)
        return images

    def convert_paths_to_nd_array(self, paths, num):
        images = []
        for path in paths:
            img = self.parse_img(path, num, 128)
            images.extend(img)
        images = np.concatenate(images)
        return images

    def build_data(self, epochs, batch_size, shuffle=True):
        for epoch in range(epochs):
            for batch_x, batch_y in self.batch_iter(self.data, self.label, batch_size, shuffle):
                yield self.convert_paths_to_nd_array(batch_x, 4), self.convert_paths_to_nd_array(batch_y, 4)
        pass

    @staticmethod
    def convert_img(img_dirs):

        def parse_img(img_dir):
            img = cv.imread(img_dir) / 255.0
            img = cv.resize(img, (128, 128))
            img = img.reshape(-1, 128, 128, 3)
            return img

        images = [parse_img(img_dir) for img_dir in img_dirs]
        return np.concatenate(images)

    @staticmethod
    def batch_iter(data, labels, batch_size, shuffle=True):
        data_size = len(data)
        data = np.array(data)
        labels = np.array(labels)
        num_batches = ((data_size - 1) // batch_size) + 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = data[shuffle_indices]
            shuffled_y = labels[shuffle_indices]
        else:
            shuffled_x = data
            shuffled_y = labels
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]

    @staticmethod
    def plot_img(images, count):
        image = np.zeros([512, 512, 3])
        for height in range(4):
            for width in range(4):
                image[height * 128:height * 128 + 128, width * 128:width * 128 + 128, :] = images[4 * height + width]
        image_decode = image * 255
        cv.imwrite('../images/result/' + str(count) + '.jpg', image_decode)


def main():
    data = SuperResolutionData()
    for batch_x, batch_y in data.build_data(1, 2):
        print(batch_x.shape, batch_y.shape)


if __name__ == '__main__':
    main()
