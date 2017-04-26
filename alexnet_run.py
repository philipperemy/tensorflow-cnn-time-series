import os

import numpy as np
import tensorflow as tf

from alexnet_data import read_dataset, next_batch, compute_mean, subtract_mean
from generate_fake_data import DATA_FOLDER

if __name__ == '__main__':

    NUM_TRAINING_IMAGES = 1000
    NUM_TESTING_IMAGES = 1000

    names = os.listdir(os.path.join(DATA_FOLDER, 'train'))
    print(names)
    NUM_CLASSES = len(names)
    class_mapper = {names[0]: 0.0, names[1]: 1.0}
    print(class_mapper)
    BATCH_SIZE = 128
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3
    LEARNING_RATE = 0.01
    data_percentage = 1
    num_training_images = data_percentage * NUM_TRAINING_IMAGES
    num_testing_images = data_percentage * NUM_TESTING_IMAGES

    print('read_dataset() start')
    training_inputs, testing_inputs = read_dataset(DATA_FOLDER, num_training_images, num_testing_images, class_mapper)
    print('read_dataset() done')
    print('compute_mean() start')
    mean_image = compute_mean(training_inputs)
    print('compute_mean() done')
    training_inputs = subtract_mean(training_inputs, mean_image)
    testing_inputs = subtract_mean(testing_inputs, mean_image)
    print(len(training_inputs), 'training inputs')
    print(len(testing_inputs), 'testing inputs')

    x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNELS])
    y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32)

    from alexnet_keras import alex_net_keras

    logits = alex_net_keras(x, num_classes=len(names), keep_prob=keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(int(1e9)):
        batch_xs, batch_ys, _ = next_batch(training_inputs, i, BATCH_SIZE)
        tr_loss, _ = sess.run([cross_entropy, train_step],
                              feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        print('[TRAINING] #batch = {0}, tr_loss = {1:.3f}'.format(i, tr_loss))
        if i % 100 == 0:
            accuracy_list = []
            j = 0
            while True:
                batch_xt, batch_yt, reset = next_batch(testing_inputs, j, BATCH_SIZE)
                if reset:
                    break
                te_loss, te_acc = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: batch_xt, y: batch_yt, keep_prob: 1.0})
                print('[TESTING] #batch = {0}, te_loss = {1:.3f}, te_acc = {2:.3f}'.format(i, te_loss, te_acc))
                accuracy_list.append(te_acc)
                j += 1
            print('[ALL] total batches = {0} total mean accuracy on testing set = {1:.2f}'.format(i, np.mean(
                accuracy_list)))
