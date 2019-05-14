import tensorflow as tf

from models.data.super_resolution_data import SuperResolutionData
from models.models.model import SuperResolutionNet


def main():
    model = SuperResolutionNet((128, 128, 3), (128, 128, 3), 0.0002, 0.7)
    data = SuperResolutionData()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    model_path = '../tmp/model.ckpt'

    # saver.restore(sess, model_path)
    # print("Model restored.")

    epochs = 200
    count = 0
    print('start training')
    for epoch in range(epochs):
        for batch_x, batch_y in data.build_data(1, 5):
            model_dict = {model.inputs: batch_x, model.outputs: batch_y}
            sess.run([model.opt], feed_dict=model_dict)
            count += 1
            if count % 10 == 0:
                loss, decode = sess.run([model.loss, model.decode], feed_dict=model_dict)
                data.plot_img(decode[:16], count // 20)
                print('Epoch:{}/{}'.format(epoch, epochs),
                      'Train count: {}'.format(count),
                      'Train losses: {:.4f}'.format(loss))
            # if count % 50 == 0:
                # save_path = saver.save(sess, model_path)
                # print("Model saved in path: %s" % save_path)
        model.lr = model.lr * 0.5
        print("change lr to : %s" % model.lr)
    sess.close()


if __name__ == '__main__':
    main()
