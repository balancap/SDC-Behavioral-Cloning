"""Keras Behavioral Cloning model.
"""
import h5py
import json

import cv2
import math
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix

from image_preprocessing import ImageDataGenerator


# General parameters.
BATCH_SIZE = 128
NB_EPOCHS = 20
SEED = 4242

# Image dimensions
IMG_ROWS, IMG_COLS = 160, 320
IMG_CHANNELS = 3


# ============================================================================
# Load data
# ============================================================================
def load_npz(filename, split=0.9):
    """Load data from Numpy .npz file and rescale images to [0, 1].
    Args:
      filename: dataset filename.
      split: Split proportion between train / validation datasets.
    Return:
      (X_train, y_train, X_test, y_test) Numpy arrays.
    """
    data = np.load(filename)
    images = data['images'].astype(np.float32) / 255.
    images = 2. * images - 1.

    # Angle data.
    angle = data['angle_sth32']

    # Split datasets.
    idxes = np.arange(images.shape[0])
    np.random.shuffle(idxes)
    idx = int(images.shape[0] * split)
    return (images[idxes[:idx]], angle[idxes[:idx]],
            images[idxes[idx:]], angle[idxes[idx:]])


def load_hdf5(filename, split=0.9):
    """Load data from HDF5 file and rescale images to [0, 1].
    Disclaimer: HDF5 dataset + Keras ImageDataGenerator do not seem to
    go very well together...

    Args:
      filename: dataset filename.
      split: Split proportion between train / validation datasets.
    Return:
      (X_train, y_train, X_test, y_test) Keras HDF5Matrix.
    """
    # Shape and split index
    with h5py.File(filename, 'r') as f:
        shape = f['images'].shape
        idx = int(shape[0] * split)

    def normalizer_fct(x):
        return np.divide(np.float32(x), 255.)

    # HDF5Matrix numpy style arrays.
    X_train = HDF5Matrix(filename, 'images', start=0, end=idx,
                         normalizer=normalizer_fct)
    y_train = HDF5Matrix(filename, 'angle', start=0, end=idx)
    X_test = HDF5Matrix(filename, 'images', start=idx, end=None,
                        normalizer=normalizer_fct)
    y_test = HDF5Matrix(filename, 'angle', start=idx, end=None)

    return (X_train, y_train, X_test, y_test)


# ============================================================================
# Model and training
# ============================================================================
def cnn_model(shape):
    """Create the model learning the behavioral cloning from driving data.
    Inspired by NVIDIA paper on this topic.
    """
    model = Sequential()

    # First 5x5 convolutions layers.
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2),
                            border_mode='valid',
                            input_shape=shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2),
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2),
                            border_mode='valid'))
    model.add(Activation('relu'))

    # 3x3 Convolutions.
    model.add(Convolution2D(64, 3, 3,
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3,
                            border_mode='valid'))
    model.add(Activation('relu'))

    # Flatten + FC layers.
    model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    return model


def train_model(filename):
    # Load dataset.
    (X_train, y_train, X_test, y_test) = load_npz(filename, split=0.9)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print('X_train shape:', X_train.shape)

    # CNN Model.
    model = cnn_model(X_train.shape[1:])

    # Train the model using SGD + momentum.
    optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mean_absolute_error'])

    # Pre-processing and realtime data augmentation.
    datagen = ImageDataGenerator(
        featurewise_center=False,   # Input mean to 0 over dataset.
        samplewise_center=True,    # Each sample mean to 0.
        featurewise_std_normalization=False,  # Divide inputs by STD of the dataset.
        samplewise_std_normalization=True,   # Divide each input by its STD.
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
    ]

    model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=BATCH_SIZE,
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
    filename = './data/4/dataset.npz'
    train_model(filename)


if __name__ == '__main__':
    main()
