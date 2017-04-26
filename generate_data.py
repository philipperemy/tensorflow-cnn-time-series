import errno
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_FOLDER = '/tmp/cnn-time-series/'


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def generate_time_series(arr, filename):
    fig = plt.figure()
    plt.plot(arr)
    plt.savefig(filename)
    plt.close(fig)


def generate():
    for i in range(200):
        if i % 2 == 0:
            direction = 'UP'
        else:
            direction = 'DOWN'
        train_output_dir = os.path.join(DATA_FOLDER, 'train', direction)
        mkdir_p(train_output_dir)
        arr = np.cumsum(np.random.standard_normal(1024))
        generate_time_series(arr, os.path.join(train_output_dir, 'img_{}.png'.format(i)))

        test_output_dir = os.path.join(DATA_FOLDER, 'test', direction)
        mkdir_p(test_output_dir)
        # arr = np.cumsum(np.random.standard_normal(1024))
        generate_time_series(arr, os.path.join(test_output_dir, 'img_{}.png'.format(i)))


if __name__ == '__main__':
    generate()
