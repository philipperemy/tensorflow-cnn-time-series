from alexnet_data import *


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
