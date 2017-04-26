import shutil

import progressbar
import scipy.stats as stats

from alexnet_data import *


def generate_two_correlated_time_series(size, rho):
    num_samples = size
    num_variables = 2
    cov = [[1.0, rho], [rho, 1.0]]

    L = np.linalg.cholesky(cov)

    uncorrelated = np.random.standard_normal((num_variables, num_samples))
    correlated = np.dot(L, uncorrelated)
    x, y = correlated
    rho, p_val = stats.pearsonr(x, y)
    return x, y, rho


def generate():
    shutil.rmtree(DATA_FOLDER, ignore_errors=True)
    size = 1024
    total_num_images = 100000
    bar = progressbar.ProgressBar()
    for i in bar(range(total_num_images)):
        if i % 2 == 0:
            class_name = 'CORRELATED'
            rho = 0.8
        else:
            class_name = 'UNCORRELATED'
            rho = 0.0

        # more fun to consider non-stationary time series.
        # still the correlation holds almost surely.
        x_tr, y_tr, _ = generate_two_correlated_time_series(size, rho)
        x_tr = np.cumsum(x_tr)
        y_tr = np.cumsum(y_tr)

        x_te, y_te, _ = generate_two_correlated_time_series(size, rho)
        x_te = np.cumsum(x_te)
        y_te = np.cumsum(y_te)

        train_output_dir = os.path.join(DATA_FOLDER, 'train', class_name)
        mkdir_p(train_output_dir)
        generate_multi_time_series([x_tr, y_tr], os.path.join(train_output_dir, '{}.png'.format(i)))

        test_output_dir = os.path.join(DATA_FOLDER, 'test', class_name)
        mkdir_p(test_output_dir)
        generate_multi_time_series([x_te, y_te], os.path.join(test_output_dir, '{}.png'.format(i)))


if __name__ == '__main__':
    generate()
