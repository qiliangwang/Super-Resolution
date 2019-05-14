import tensorflow as tf


class SuperResolutionNet(object):
    def __init__(self, input_dim, out_dim, lr, drop_out):
        self.inputs = tf.placeholder(tf.float32, [None, *input_dim], name='inputs')
        self.outputs = tf.placeholder(tf.float32, [None, *out_dim], name='outputs')
        self.lr = lr
        self.drop = drop_out
        self.compressed = self.decoder()
        self.logits, self.decode = self.encoder()
        self.loss = self.create_loss()
        self.opt = self.optimizer()
        pass

    def decoder(self):
        with tf.variable_scope('decoder'):
            print(self.inputs)
            x1 = tf.layers.conv2d(self.inputs, 64, kernel_size=[7, 7], strides=[1, 1], padding='same')
            print(x1)

            x2 = tf.layers.conv2d(x1, 64, kernel_size=[7, 7], strides=[1, 1], padding='same',
                                  activation=tf.nn.leaky_relu)
            print(x2)

            x3 = tf.layers.conv2d(x2, 64, kernel_size=[7, 7], strides=[1, 1], padding='same',
                                  activation=tf.nn.leaky_relu)
            print(x3)

            # x4 = tf.layers.conv2d(x3, 512, kernel_size=[7, 7], strides=[1, 1], padding='same',
            #                       activation=tf.nn.leaky_relu)
            # print(x4)

            x5 = tf.layers.conv2d(x3, 64, kernel_size=[7, 7], strides=[1, 1], padding='same',
                                  activation=tf.nn.leaky_relu)
            print(x5)

            return x5
        pass

    def encoder(self):
        with tf.variable_scope('encoder'):
            # x4 = tf.layers.conv2d_transpose(self.compressed, 512, kernel_size=[7, 7], strides=[1, 1], padding='same',
            #                                 activation=tf.nn.leaky_relu)
            # print(x4)

            x3 = tf.layers.conv2d_transpose(self.compressed, 64, kernel_size=[7, 7], strides=[1, 1], padding='same',
                                            activation=tf.nn.leaky_relu)
            print(x3)

            x2 = tf.layers.conv2d_transpose(x3, 64, kernel_size=[7, 7], strides=[1, 1], padding='same',
                                            activation=tf.nn.leaky_relu)
            print(x2)

            x1 = tf.layers.conv2d_transpose(x2, 64, kernel_size=[7, 7], strides=[1, 1], padding='same',
                                            activation=tf.nn.leaky_relu)
            print(x1)

            logits = tf.layers.conv2d_transpose(x1, 3, kernel_size=[7, 7], strides=[1, 1], padding='same')

            decode = tf.nn.sigmoid(logits, name='decode')

            return logits, decode
    #
    # def encoder(self):
    #     with tf.variable_scope('encoder'):
    #
    #         x4 = tf.layers.conv2d_transpose(self.compressed, 512, kernel_size=[7, 7], strides=[1, 1], padding='same',
    #                                         activation=tf.nn.leaky_relu)
    #         print(x4)
    #
    #         x3 = tf.layers.conv2d_transpose(x4, 256, kernel_size=[7, 7], strides=[1, 1], padding='same',
    #                                         activation=tf.nn.leaky_relu)
    #         print(x3)
    #
    #         x2 = tf.layers.conv2d_transpose(x3, 128, kernel_size=[7, 7], strides=[1, 1], padding='same',
    #                                         activation=tf.nn.leaky_relu)
    #         print(x2)
    #
    #         x1 = tf.layers.conv2d_transpose(x2, 64, kernel_size=[7, 7], strides=[1, 1], padding='same',
    #                                         activation=tf.nn.leaky_relu)
    #         print(x1)
    #
    #         logits = tf.layers.conv2d_transpose(x1, 3, kernel_size=[7, 7], strides=[1, 1], padding='same')
    #
    #         decode = tf.nn.sigmoid(logits, name='decode')
    #
    #         return logits, decode

    def create_loss(self):
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.outputs, logits=self.logits)
        loss = tf.reduce_mean(cost)
        return loss

    def optimizer(self):
        return tf.train.AdamOptimizer(0.0002).minimize(self.loss)


def main():
    model = SuperResolutionNet((64, 64, 3), (64, 64, 3), 0.1, 0.7)
    pass


if __name__ == '__main__':
    main()