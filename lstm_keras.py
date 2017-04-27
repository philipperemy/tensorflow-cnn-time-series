import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

from generate_corr_data import generate_two_correlated_time_series

max_len = 1024
batch_size = 512


def next_batch():
    x = []
    y = []
    for i in range(batch_size // 2):
        x.append(generate_two_correlated_time_series(size=max_len, rho=0.8)[0:2])
        y.append(1.0)

        x.append(generate_two_correlated_time_series(size=max_len, rho=0)[0:2])
        y.append(0.0)
    return np.transpose(np.array(x), (0, 2, 1)), np.array(y)


print('Build model...')
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, batch_input_shape=(batch_size, max_len, 2),
               return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
while True:
    x_train, y_train = next_batch()
    train_loss, train_acc = model.train_on_batch(x_train, y_train)
    print('[train] loss= {0:.3f}, acc= {1:.2f}'.format(train_loss, train_acc * 100))

    x_test, y_test = next_batch()
    test_loss, test_acc = model.test_on_batch(x_test, y_test)
    print('[test] loss= {0:.3f}, acc= {1:.2f}'.format(test_loss, test_acc * 100))
