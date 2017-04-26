import tensorflow as tf
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten


# import tensorflow as tf
# tf.python.control_flow_ops = tf  # some hack to get tf running with Dropout

# 224x224
# https://gist.github.com/JBed/c2fb3ce8ed299f197eff
def alex_net_keras(x, num_classes=2, keep_prob=0.5):
    x = Conv2D(92, kernel_size=(11, 11), strides=(4, 4), padding='same')(x)  # conv 1
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # LRN is missing here - Caffe.
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # pool 1

    x = Conv2D(256, kernel_size=(5, 5), padding='same')(x)  # miss group and pad param # conv 2
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # pool 2

    x = Conv2D(384, kernel_size=(3, 3), padding='same')(x)  # conv 3
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(384, kernel_size=(3, 3), padding='same')(x)  # conv 4
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)  # conv 5
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(4096, kernel_initializer='normal')(x)  # fc6
    # dropout 0.5
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(4096, kernel_initializer='normal')(x)  # fc7
    # dropout 0.5
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    # x = BatchNormalization()(x)
    # x = Activation('softmax')(x)
    return x
