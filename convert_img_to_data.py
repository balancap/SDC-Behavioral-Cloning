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

        # if data[i] != 0.0:
        cum_conv += data[i]
        cum_vol += 1
        # if cum_vol > 0.0:
        r[i] = cum_conv / scale
    return r


def sinc(alpha):
    y = np.ones_like(alpha)
    mask = alpha != 0.0
    y[mask] = np.sin(alpha[mask]) / alpha[mask]
    return y


def cosc(alpha):
    y = np.zeros_like(alpha)
    mask = alpha != 0.0
    y[mask] = (1. - np.cos(alpha[mask])) / alpha[mask]
    return y


def trajectory(dt, speed, angle):
    length = 5.85
    # speed = 48.28032 * 1000 / 3600

    # Rotation radius
    # radius = length / np.sin(angle)
    alpha = speed * dt / length * np.sin(angle)

    # dx displacement vectors.
    dx = np.zeros(shape=(len(angle), 2), dtype=np.float32)
    dx[:, 0] = speed * dt * cosc(alpha)
    dx[:, 1] = speed * dt * sinc(alpha)
    # dx[:, 2] = 1.0

    # Affine transformation.
    rot_trans = np.zeros(shape=(len(angle), 2, 2), dtype=np.float32)
    rot_trans[:, 0, 0] = np.cos(alpha)
    rot_trans[:, 1, 1] = np.cos(alpha)
    rot_trans[:, 0, 1] = np.sin(alpha)
    rot_trans[:, 1, 0] = -np.sin(alpha)

    # Compute position vector of the car.
    x = np.zeros(shape=(len(angle)+1, 2), dtype=np.float32)
    v1 = np.array([1., 0.], dtype=np.float32)
    v2 = np.array([0., 1.], dtype=np.float32)

    for i in range(len(angle)):
        v1 = np.matmul(rot_trans[i], v1)
        v2 = np.matmul(rot_trans[i], v2)

        x[i+1] = x[i]
        x[i+1] += v1 * dx[i, 0]
        x[i+1] += v2 * dx[i, 1]

    return x


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
                'dt': np.zeros((nb_imgs, ), dtype=np.float32),
            }

    # Load CSV information file.
    with open(path + 'driving_log.csv', 'r') as f:
        creader = csv.reader(f)
        csv_list = list(creader)

        # Load image, when exists, and associated data.
        idx_subsample = 0
        for i in range(len(csv_list)-1):
            a = csv_list[i]

            # Time difference: ugly hack!
            p0 = csv_list[i][0][:-4].split("_")
            p1 = csv_list[i+1][0][:-4].split("_")
            t0 = float(p0[-1]) * 0.001 + float(p0[-2]) + float(p0[-3]) * 60. + float(p0[-4]) * 3600.
            t1 = float(p1[-1]) * 0.001 + float(p1[-2]) + float(p1[-3]) * 60. + float(p1[-4]) * 3600.

            dt = t1 - t0

            # Open file.
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
                    data['speed'][j] = float(a[6]) * 1.609344 / 3.6
                    data['dt'][j] = dt

                idx_subsample += 1
        print('')

    # Compute trajectory.
    data['x'] = trajectory(data['dt'], data['speed'], data['angle'])

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
    path = './data/50hz_1/'
    print('Dataset path: ', path)

    # Load data and 'pickle' dump.
    data = load_data(path, mask=True)
    save_np_data(path, data)
    # dump_data(path, data)

    # HDF5 dataset.
    # create_hdf5(path)

if __name__ == '__main__':
    main()
