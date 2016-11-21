"""Convert IMAGES and CSV to pickle data files.
"""
import os
import sys

import pickle
import csv

import cv2
import math
import numpy as np

import matplotlib.image as mpimg

IMG_SHAPE = (160, 320, 3)
SUBSAMPLING = 2


def load_data(path):
    """Load DATA from path: images + car information.

    Return:
        Dictionary containing the following entries:
         - images;
         - angle: steering angle;
         - throttle;
         - speed;
    """
    list_imgs = os.listdir(path + 'IMG/')
    print('Number of images: %i. Sub-sampling: %i.' % (len(list_imgs),
                                                       SUBSAMPLING))

    nb_imgs = math.ceil(len(list_imgs) / float(SUBSAMPLING))
    data = {
                'images': np.zeros((nb_imgs, *IMG_SHAPE), dtype=np.uint8),
                'angle': np.zeros((nb_imgs, ), dtype=np.float32),
                'throttle': np.zeros((nb_imgs, ), dtype=np.float32),
                'speed': np.zeros((nb_imgs, ), dtype=np.float32),
            }

    # Load CSV information file.
    with open(path + 'driving_log.csv', 'r') as f:
        creader = csv.reader(f)
        csv_list = list(creader)

        # Load image, when exists, and associated data.
        idx_subsample = 0
        for i, a in enumerate(csv_list):
            filename = a[0]
            if os.path.isfile(filename):
                idx_subsample += 1
                if idx_subsample % SUBSAMPLING == 0:
                    j = idx_subsample // SUBSAMPLING
                    sys.stdout.write('\r>> Converting image %d/%d' % (j+1, nb_imgs))
                    sys.stdout.flush()

                    # Copy data.
                    data['images'][j] = mpimg.imread(filename)
                    data['angle'][j] = float(a[3])
                    data['throttle'][j] = float(a[4])
                    data['speed'][j] = float(a[6])
        print('')
        return data


def dump_data(path, data):
    """Dump data using Pickle.
    """
    filename = path + 'data.p'
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)


def save_np_data(path, data):
    """Save data as npz file.
    """
    filename = path + 'data.npz'
    np.savez(filename, **data)


def main():
    path = './data/1/'

    # Load data and 'pickle' dump.
    data = load_data(path)
    save_np_data(path, data)
    # dump_data(path, data)

if __name__ == '__main__':
    main()
