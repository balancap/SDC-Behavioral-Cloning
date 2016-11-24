"""Convert IMAGES and CSV to pickle data files.
"""
import os
import sys

import pickle
import csv
import h5py
import numba

import cv2
import math
import numpy as np

import matplotlib.image as mpimg

IMG_SHAPE = (160, 320, 3)
SUBSAMPLING = 1

MASK_PRE_FRAMES = 0
MASK_POST_FRAMES = 0


# ============================================================================
# Tools
# ============================================================================
@numba.jit(nopython=True, cache=True)
def np_exp_conv(data, scale):
    """Exponential convolution.

    :param data: Numpy array of transactions volume.
    :param scale: Exponential scaling parameter
    """

    N = data.size
    r = np.zeros(N, np.float64)
    if N == 0:
        return r

    cum_conv = 0.0
    cum_vol = 0.0
    for i in range(N):
        if i >= 1:
            cum_conv = cum_conv * np.exp(-1. / scale)
            cum_vol = cum_vol * np.exp(-1. / scale)

        cum_conv += data[i]
        cum_vol += 1
        r[i] = cum_conv / cum_vol
    return r


# ============================================================================
# Load / Save data: old way!
# ============================================================================
def load_data(path, mask=True):
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
    # Data structure.
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
                if idx_subsample % SUBSAMPLING == 0:
                    j = idx_subsample // SUBSAMPLING
                    sys.stdout.write('\r>> Converting image %d/%d' % (j+1, nb_imgs))
                    sys.stdout.flush()

                    # Copy data.
                    data['images'][j] = mpimg.imread(filename)
                    data['angle'][j] = float(a[3])
                    data['throttle'][j] = float(a[4])
                    data['speed'][j] = float(a[6])
                idx_subsample += 1
        print('')

    # Post-processing angle: exponential smoothing.
    scales = [1., 2., 4., 8., 16., 32.]
    for s in scales:
        data['angle_sth%i' % s] = np_exp_conv(data['angle'], s)
        data['angle_rsth%i' % s] = np_exp_conv(data['angle'][::-1], s)[::-1]

    # Post-processing: pre-angle.
    scales = [2, 3, 4, 6]
    for s in scales:
        data['angle_pre%i' % s] = np.zeros_like(data['angle'])
        for i in range(len(data['angle'])):
            data['angle_pre%i' % s][i-s+1:i+1] += data['angle'][i] / s

    # Mask data: keep frames after turning event only (1 frame ~ 0.1 second).
    if mask:
        shape = data['images'].shape
        mask = np.zeros((shape[0], ), dtype=bool)
        for i in range(shape[0]):
            if data['angle'][i] != 0.:
                mask[i-MASK_PRE_FRAMES:i+MASK_POST_FRAMES+1] = True
        # Apply mask.
        for k in data.keys():
            data[k] = data[k][mask]

    return data


def dump_data(path, data):
    """Dump data using Pickle.
    """
    filename = path + 'dataset.p'
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)


def save_np_data(path, data):
    """Save data as npz file.
    """
    filename = path + 'dataset.npz'
    np.savez(filename, **data)


# ============================================================================
# Load / Save data: HDF5 file.
# ============================================================================
def create_hdf5(path):
    """Create HDF5 file from images and CSV data on the disk.

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

    with h5py.File(path + 'dataset.hdf5', 'w') as f, \
            open(path + 'driving_log.csv', 'r') as fcsv:
        # Create HDF5 dataset.
        images = f.create_dataset('images', (nb_imgs, *IMG_SHAPE), dtype='uint8')
        angle = f.create_dataset('angle', (nb_imgs, ), dtype='float32')
        throttle = f.create_dataset('throttle', (nb_imgs, ), dtype='float32')
        speed = f.create_dataset('speed', (nb_imgs, ), dtype='float32')

        # Load CSV information file.
        creader = csv.reader(fcsv)
        csv_list = list(creader)

        # Load image, when exists, and associated data.
        idx_subsample = 0
        for a in csv_list:
            filename = a[0]
            if os.path.isfile(filename):
                if idx_subsample % SUBSAMPLING == 0:
                    j = idx_subsample // SUBSAMPLING
                    sys.stdout.write('\r>> Converting image %d/%d' % (j+1,
                                                                      nb_imgs))
                    sys.stdout.flush()

                    # Save data to HDF5.
                    images[j] = mpimg.imread(filename)
                    angle[j] = float(a[3])
                    throttle[j] = float(a[4])
                    speed[j] = float(a[6])
                idx_subsample += 1
        print('')

        # Post processing angle: exponential smoothing.
        scales = [1., 2., 4., 8., 16., 32., 64., 128.]
        for s in scales:
            angle_smooth = f.create_dataset('angle_sth%i' % s, (nb_imgs, ),
                                            dtype='float32')
            angle_smooth[:] = np_exp_conv(angle[()], s)


def main():
    path = './data/7/'
    print('Dataset path: ', path)

    # Load data and 'pickle' dump.
    data = load_data(path, mask=False)
    save_np_data(path, data)
    # dump_data(path, data)

    # HDF5 dataset.
    # create_hdf5(path)

if __name__ == '__main__':
    main()
