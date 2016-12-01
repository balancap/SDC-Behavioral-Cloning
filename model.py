"""Keras Behavioral Cloning model.
"""
import h5py
import json

import cv2
import math
import numpy as np

import keras
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix

from image_preprocessing import ImageDataGenerator


# General parameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DECAY = 5e-5
BN_EPSILON = 1e-6
NB_EPOCHS = 100

SEED = 4242

# Image dimensions
IMG_ROWS, IMG_COLS = 160, 320
IMG_ROWS, IMG_COLS = 95, 320
IMG_CHANNELS = 3


# ============================================================================
# Load data
# ============================================================================
def load_npz(filenames, split=0.9, angle_key='angle'):
    """Load data from Numpy .npz files and rescale images to [0, 1].
    Args:
      filenames: List of dataset filenames.
      split: Split proportion between train / validation datasets.
    Return:
      (X_train, y_train, X_test, y_test) Numpy arrays.
    """
    # Load data from numpy files.
    images = None
    angle = None
    for path in filenames:
        data = np.load(path)
        if images is None:
            images = data['images'].astype(np.float32) * 2 / 255. - 1.
            angle = data[angle_key]
        else:
            images = np.append(images, data['images'].astype(np.float32) * 2 / 255. - 1.,
                               axis=0)
            angle = np.append(angle, data[angle_key], axis=0)

    # Pre-process images.
    # images *= 2

    # Angle translation.
    # delta = 6
    # angle = angle[delta:]
    # angle = np.lib.pad(angle, ((0, delta)), 'symmetric')

    # Shuffle and Split datasets.
    idxes = np.arange(images.shape[0])
    np.random.shuffle(idxes)
    idx = int(images.shape[0] * split)

    return (images[idxes[:idx]], angle[idxes[:idx]],
            images[idxes[idx:]], angle[idxes[idx:]])

    # return (images, angle,
    #         images[idxes[idx:]], angle[idxes[idx:]])


# ============================================================================
# Model and training
# ============================================================================
def cnn_model(shape):
    """Create the model learning the behavioral cloning from driving data.
    Inspired by NVIDIA paper on this topic.
    """
    l2_weight = 0.0001
    model = Sequential()

    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999, input_shape=shape))
    # First 5x5 convolutions layers.
    model.add(Convolution2D(24, 5, 5,
                            subsample=(2, 2),
                            # init='normal',
                            # input_shape=shape,
                            border_mode='valid'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999))
    model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
    print('Layer 1: ', model.layers[-1].output_shape)

    model.add(Convolution2D(36, 5, 5,
                            subsample=(2, 2),
                            # init='normal',
                            border_mode='valid'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    print('Layer 2: ', model.layers[-1].output_shape)

    # model.add(Convolution2D(48, 5, 5,
    #                         subsample=(2, 2),
    #                         # init='normal',
    #                         border_mode='valid'))
    # model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3), strides=None, border_mode='valid'))
    # print('Layer 3: ', model.layers[-1].output_shape)

    model.add(Convolution2D(54, 5, 5,
                            subsample=(2, 2),
                            # init='normal',
                            border_mode='valid'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    print('Layer 3: ', model.layers[-1].output_shape)

    # 3x3 Convolutions.
    model.add(Convolution2D(64, 3, 3,
                            # init='normal',
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999))
    print('Layer 4: ', model.layers[-1].output_shape)

    model.add(Convolution2D(80, 3, 3,
                            # init='normal',
                            border_mode='valid'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999))
    model.add(Activation('relu'))
    print('Layer 5: ', model.layers[-1].output_shape)

    model.add(Convolution2D(96, 3, 3,
                            # init='normal',
                            border_mode='valid'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=0.999))
    model.add(Activation('relu'))
    print('Layer 6: ', model.layers[-1].output_shape)

    # Flatten + FC layers.
    model.add(Flatten())
    # model.add(Dense(1000))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(100, W_regularizer=l2(l2_weight)))
    # model.add(BatchNormalization(mode=1, epsilon=BN_EPSILON, momentum=0.999))
    # model.add(Activation('relu'))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
    model.add(Dropout(0.5))

    model.add(Dense(50, W_regularizer=l2(l2_weight)))
    # model.add(BatchNormalization(mode=1, epsilon=BN_EPSILON, momentum=0.999))
    # model.add(Activation('relu'))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))

    model.add(Dense(10, W_regularizer=l2(l2_weight)))
    # model.add(BatchNormalization(mode=1, epsilon=BN_EPSILON, momentum=0.999))
    # model.add(Activation('relu'))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.0))

    model.add(Dense(1))
    return model


def train_model(X_train, y_train, X_test, y_test):
    # Training information.
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print('X_train shape:', X_train.shape)

    # Training weights: more on large angles.
    # y_weights = 1. + 20. * np.abs(y_train)
    y_weights = np.ones_like(y_train)

    # CNN Model.
    model = cnn_model(X_train.shape[1:])

    # Train the model using SGD + momentum.
    optimizer = SGD(lr=0.1,
                    decay=1e-6,
                    momentum=0.9,
                    nesterov=True)
    optimizer = keras.optimizers.RMSprop(lr=LEARNING_RATE,
                                         rho=0.9,
                                         epsilon=1e-08,
                                         decay=DECAY)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE,
                                      beta_1=0.9, beta_2=0.999,
                                      epsilon=1e-08,
                                      decay=DECAY)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mean_absolute_error'])

    # Pre-processing and realtime data augmentation.
    datagen = ImageDataGenerator(
        featurewise_center=False,   # Input mean to 0 over dataset.
        samplewise_center=False,    # Each sample mean to 0.
        featurewise_std_normalization=False,  # Divide inputs by STD of the dataset.
        samplewise_std_normalization=False,   # Divide each input by its STD.
        zca_whitening=False,        # Apply ZCA whitening
        rotation_range=0,           # Randomly rotate images.
        width_shift_range=0.,       # Random shift (fraction of total width).
        height_shift_range=0.,      # Random shift (fraction of total height).
        horizontal_flip=True,       # Random horizontal flip.
        vertical_flip=False)        # Random vertical flip.

    # Compute quantities required for featurewise normalization.
    # (std, mean, and principal components if ZCA whitening is applied)
    # datagen.fit(X_train)

    # Fit the model with batches generated by datagen.flow()
    callbacks = [
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                    write_graph=True, write_images=True)
        # keras.callbacks.LearningRateScheduler(lamda:x x)
    ]

    model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=BATCH_SIZE,
                                     sample_weight=y_weights,
                                     # save_to_dir='./img/',
                                     # save_format='png',
                                     shuffle=True),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=NB_EPOCHS,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        max_q_size=10,
                        nb_worker=1,
                        pickle_safe=False)

    # model.fit(X_train, y_train, batch_size=32, shuffle='batch')
    # Save model parameters and arch.
    model.save('model.h5')
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

    # with open('model.json', 'w') as f:
    #     json.dump(json.loads(model.to_json()), f,
    #               indent=4, separators=(',', ': '))


def main():
    np.random.seed(SEED)
    filenames = [
                 # './data/3/dataset.npz',
                 # './data/4/dataset.npz',
                 './data/q3_recover_left/dataset.npz',
                 './data/q3_recover_right/dataset.npz',
                 './data/q3_recover_left2/dataset.npz',
                 './data/q3_recover_right2/dataset.npz',
                 './data/q3_clean/dataset.npz',
                 './data/q3_clean2/dataset.npz',
                 # './data/5/dataset.npz'
                 ]
                 # './data/5/dataset.npz']
    # filenames = ['./data/7/dataset.npz',
    #              './data/8/dataset.npz']
#     filenames = ['./data/1/dataset.npz']
    # filenames = ['./data/50hz_1/dataset.npz']
    # filenames = [
    #              './data/q3_clean/dataset.npz',
    #              './data/q3_recover_left/dataset.npz',
    #              './data/q3_recover_right/dataset.npz',
    #             ]

    # Load dataset.
    (X_train, y_train, X_test, y_test) = load_npz(filenames,
                                                  split=0.9,
                                                  angle_key='angle_med6')
    # train model.
    train_model(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
