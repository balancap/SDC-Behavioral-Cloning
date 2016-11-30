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

# IMG_SHAPE = (160, 320, 3)
IMG_SHAPE = (95, 320, 3)
SUBSAMPLING = 1

MASK_PRE_FRAMES = 3
MASK_POST_FRAMES = 0

CAR_LENGTH = 5.9
CAR_OFFSET = 1.0


def image_preprocessing(img):
    # Cut bottom and top.
    out = img
    out = out[50:-15, :, :]
    # out = cv2.resize(out, (IMG_SHAPE[1], IMG_SHAPE[0]), interpolation=cv2.INTER_LANCZOS4)
    # out = out[34:-10, :, :]
    # out = cv2.cvtColor(out, cv2.COLOR_BGR2HLS)
    return out

# ============================================================================
# Numpy Tools
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


def mask_nonzero(data):
    mask = np.zeros((data.shape[0], ), dtype=bool)
    for i in range(data.shape[0]):
        if data[i] != 0.:
            mask[i-MASK_PRE_FRAMES:i+MASK_POST_FRAMES+1] = True
    return mask


def mask_positive(data):
    mask = np.zeros((data.shape[0], ), dtype=bool)
    for i in range(data.shape[0]):
        if data[i] > 0.:
            mask[i-MASK_PRE_FRAMES:i+MASK_POST_FRAMES+1] = True
    return mask


def mask_negative(data):
    mask = np.zeros((data.shape[0], ), dtype=bool)
    for i in range(data.shape[0]):
        if data[i] < 0.:
            mask[i-MASK_PRE_FRAMES:i+MASK_POST_FRAMES+1] = True
    return mask


# ============================================================================
# Trajectory and angle estimates.
# ============================================================================
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


def trajectory(dt, speed, angle, length=CAR_LENGTH):
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
    x = np.zeros(shape=(len(angle), 2), dtype=np.float32)
    v1 = np.array([1., 0.], dtype=np.float32)
    v2 = np.array([0., 1.], dtype=np.float32)

    # unit_vectors = np.zeros(shape=(len(angle), 2), dtype=np.float32)

    for i in range(len(angle)-1):
        x[i+1] = x[i]
        x[i+1] += v1 * dx[i, 0]
        x[i+1] += v2 * dx[i, 1]

        v1 = np.matmul(rot_trans[i], v1)
        v2 = np.matmul(rot_trans[i], v2)

    return x, alpha


def angle_post(alpha, dt, speed, delta=1, offset=0.0, length=CAR_LENGTH):
    # Rotation matrices.
    rot_mat = np.zeros(shape=(len(alpha), 2, 2), dtype=np.float32)
    rot_mat[:, 1, 1] = np.cos(alpha)
    rot_mat[:, 0, 0] = np.cos(alpha)
    rot_mat[:, 0, 1] = np.sin(alpha)
    rot_mat[:, 1, 0] = -np.sin(alpha)

    # dx displacement vectors.
    dx = np.zeros(shape=(len(alpha), 2), dtype=np.float32)
    dx[:, 0] = speed * dt * cosc(alpha)
    dx[:, 1] = speed * dt * sinc(alpha)
    steps = np.ones(shape=(len(alpha), ), dtype=np.int8)

    # Local coordinate system. TODO: dense matrix notation...
    ax = np.zeros(shape=(len(alpha), 2, 1), dtype=np.float32)
    ay = np.zeros(shape=(len(alpha), 2, 1), dtype=np.float32)
    ax[:, 0, 0] = ay[:, 1, 0] = 1.0
    ax = np.matmul(rot_mat, ax)
    ay = np.matmul(rot_mat, ay)

    # Delta - Cumulative transformations and dx.
    cumul_dx = dx.copy()
    for j in range(1, delta):
        # Update cumulative dx.
        cumul_dx[:-j, 0] += dx[j:, 0] * ax[:-j, 0, 0]
        cumul_dx[:-j, 1] += dx[j:, 0] * ax[:-j, 1, 0]

        cumul_dx[:-j, 0] += dx[j:, 1] * ay[:-j, 0, 0]
        cumul_dx[:-j, 1] += dx[j:, 1] * ay[:-j, 1, 0]

        # Update local coordinate system.
        ax[:-j] = np.matmul(rot_mat[j:], ax[:-j])
        ay[:-j] = np.matmul(rot_mat[j:], ay[:-j])
        steps[:-j] += 1

    # Fit a Bezier curve! See: https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    P0 = np.zeros(shape=(len(alpha), 2), dtype=np.float32)
    P0[:, 0] = offset
    P3 = cumul_dx
    P1 = P0.copy()
    P2 = P3.copy()

    # Vertical displacement... 1/3 factor coming from speed normalisation.
    factor = 0.3333
    P1[:, 1] += dt * speed * steps * factor
    # Same for P2...
    ay = np.squeeze(ay)
    P2[:, 0] += dt * speed * steps * (-ay[:, 0]) * factor
    P2[:, 1] += dt * speed * steps * (-ay[:, 1]) * factor

    # Check control points for licit values...
    mask = (P3[:, 0] > P0[:, 0]) * (P2[:, 0] < P0[:, 0])
    P2[mask, 0] = P0[mask, 0]
    mask = (P3[:, 0] < P0[:, 0]) * (P2[:, 0] > P0[:, 0])
    P2[mask, 0] = P0[mask, 0]
    mask = (P3[:, 1] > 0.) * (P2[:, 1] < 0.)
    P2[mask, 1] = 0.0
    mask = (P3[:, 1] < 0.) * (P2[:, 1] > 0.)
    P2[mask, 1] = 0.0

    mask = (P1[:, 1] > P3[:, 1])
    P1[mask, 1] = P3[mask, 1]

    # Inverse curvature at zero and angle.
    dvx = (P1 - P0) / factor
    ddvx = 2 * (P2 - 2*P1 + P0) / factor
    kappa = -(ddvx[:, 1] * dvx[:, 0] - ddvx[:, 0] * dvx[:, 1]) / ((dvx[:, 0]**2 + dvx[:, 1]**2) ** 1.5)
    angle = np.arcsin(kappa * length)

    # Parameters in equation: ax - b = 0.
#     a = np.squeeze(ay)
#     b = a[:, 0] * cumul_dx[:, 0] + a[:, 1] * cumul_dx[:, 1]
#     # Inverse radius and angle.
#     inv_radius = a[:, 0] / ( b - a[:, 0] * offset )
#     angle = np.arcsin(inv_radius * length)

    angle[np.isnan(angle)] = 0.0

    return angle


def angle_median(alpha, dt, speed, delta=1, offset=0.0, length=CAR_LENGTH):
    # Rotation radius
    # radius = length / np.sin(angle)
    # alpha = speed * dt / length * np.sin(angle)

    # Rotation matrices.
    rot_mat = np.zeros(shape=(len(alpha), 2, 2), dtype=np.float32)
    rot_mat[:, 1, 1] = np.cos(alpha)
    rot_mat[:, 0, 0] = np.cos(alpha)
    rot_mat[:, 0, 1] = np.sin(alpha)
    rot_mat[:, 1, 0] = -np.sin(alpha)

    # dx displacement vectors.
    dx = np.zeros(shape=(len(alpha), 2), dtype=np.float32)
    dx[:, 0] = speed * dt * cosc(alpha)
    dx[:, 1] = speed * dt * sinc(alpha)
    steps = np.ones(shape=(len(alpha), ), dtype=np.int8)

    # Local coordinate system. TODO: dense matrix notation...
    ax = np.zeros(shape=(len(alpha), 2, 1), dtype=np.float32)
    ay = np.zeros(shape=(len(alpha), 2, 1), dtype=np.float32)
    ax[:, 0, 0] = ay[:, 1, 0] = 1.0
    ax = np.matmul(rot_mat, ax)
    ay = np.matmul(rot_mat, ay)

    # Delta - Cumulative transformations and dx.
    cumul_dx = dx.copy()
    for j in range(1, delta):
        # Update cumulative dx.
        cumul_dx[:-j, 0] += dx[j:, 0] * ax[:-j, 0, 0]
        cumul_dx[:-j, 1] += dx[j:, 0] * ax[:-j, 1, 0]

        cumul_dx[:-j, 0] += dx[j:, 1] * ay[:-j, 0, 0]
        cumul_dx[:-j, 1] += dx[j:, 1] * ay[:-j, 1, 0]

        # Update local coordinate system.
        ax[:-j] = np.matmul(rot_mat[j:], ax[:-j])
        ay[:-j] = np.matmul(rot_mat[j:], ay[:-j])
        steps[:-j] += 1

    # Median fit...
    P0 = np.zeros(shape=(len(alpha), 2), dtype=np.float32)
    P0[:, 0] = offset
    P1 = cumul_dx

    # Parameters in equation: ax - b = 0.
    a = P1 - P0
    m = (P0 + P1) / 2.
    b = a[:, 0] * m[:, 0] + a[:, 1] * m[:, 1]
    # Inverse radius and angle.
    kappa = a[:, 0] / b
    angle = np.arcsin(kappa * length)

    # Just in case...
    angle[np.isnan(angle)] = 0.0

    return angle


def angle_post_mean(x, alpha, dt, speed, deltas=None, length=CAR_LENGTH):
    deltas = deltas or [1]

    avg = np.zeros(shape=(len(alpha),), dtype=np.float32)
    for d in deltas:
        a = angle_post(alpha, dt, speed, delta=d, length=length)
        avg += a
    avg = avg / len(deltas)
    return avg


def angle_curvature(x, delta=1, length=CAR_LENGTH):
    def deriv(x):
        """Compute derivative with zero padding.
        """
        dx = x[2*delta:] - x[:-2*delta]
        dx = np.lib.pad(dx, ((delta, delta), (0, 0)), 'symmetric')
        return dx / 2.

    # First and second derivative...
    dvx = deriv(x)
    ddvx = deriv(dvx)
    # Inverse curvature.
    kappa = (ddvx[:, 1] * dvx[:, 0] - ddvx[:, 0] * dvx[:, 1]) / ((dvx[:, 0]**2 + dvx[:, 1]**2) ** 1.5)
    angle = np.arcsin(-kappa * length)

    # angle = angle[delta:]
    # angle = np.lib.pad(angle, ((0, delta)), 'symmetric')
    return angle


# ============================================================================
# Load / Save data: old way!
# ============================================================================
def load_data(path, fmask=None):
    """Load DATA from path: images + car information.

    Return:
        Dictionary containing the following entries:
         - images;
         - angle: steering angle;
         - throttle;
         - speed;
    """
    # List of 'center' images.
    list_imgs = os.listdir(path + 'IMG/')
    nb_img_center = len([l for l in list_imgs if 'center' in l])
    nb_img_left = len([l for l in list_imgs if 'left' in l])
    nb_img_right = len([l for l in list_imgs if 'right' in l])

    # Number of types (left-center-right) and images.
    nb_types = 1
    nb_imgs = nb_img_center - 1
    if nb_img_left == nb_img_center and nb_img_right == nb_img_center:
        nb_types = 3

    print('Number of images: %i. Categories: %i.' % (nb_imgs, nb_types))

    # Data structure.
    data = {
                'images': np.zeros((nb_types, nb_imgs, *IMG_SHAPE), dtype=np.uint8),
                'angle': np.zeros((nb_types, nb_imgs), dtype=np.float32),
                'throttle': np.zeros((nb_types, nb_imgs), dtype=np.float32),
                'speed': np.zeros((nb_types, nb_imgs), dtype=np.float32),
                'dt': np.zeros((nb_types, nb_imgs), dtype=np.float32),
            }

    # Load CSV information file.
    with open(path + 'driving_log.csv', 'r') as f:
        creader = csv.reader(f)
        csv_list = list(creader)

    # Load image, when exists, and associated data.
    j = 0
    for i in range(len(csv_list)-1):
        a = csv_list[i]

        # Time difference: ugly hack, but working!
        p0 = csv_list[i][0][:-4].split("_")
        p1 = csv_list[i+1][0][:-4].split("_")
        t0 = float(p0[-1]) * 0.001 + float(p0[-2]) + float(p0[-3]) * 60. + float(p0[-4]) * 3600.
        t1 = float(p1[-1]) * 0.001 + float(p1[-2]) + float(p1[-3]) * 60. + float(p1[-4]) * 3600.
        dt = t1 - t0

        # Open image files.
        filename_center = a[0].strip()
        filename_left = a[1].strip()
        filename_right = a[2].strip()

        if os.path.isfile(filename_center):
            sys.stdout.write('\r>> Converting image %d/%d' % (j+1, nb_imgs))
            sys.stdout.flush()

            # Image data.
            img = mpimg.imread(filename_center)
            data['images'][0, j] = image_preprocessing(img)

            # Car data.
            data['angle'][0, j] = float(a[3])
            data['throttle'][0, j] = float(a[4])
            data['speed'][0, j] = float(a[6]) * 1.609344 / 3.6  # to m/s
            data['dt'][0, j] = dt

            # Additional left and right data.
            if nb_types > 1:
                img = mpimg.imread(filename_left)
                data['images'][1, j] = image_preprocessing(img)

                data['angle'][1, j] = float(a[3])
                data['throttle'][1, j] = float(a[4])
                data['speed'][1, j] = float(a[6]) * 1.609344 / 3.6  # to m/s
                data['dt'][1, j] = dt

                img = mpimg.imread(filename_right)
                data['images'][2, j] = image_preprocessing(img)

                data['angle'][2, j] = float(a[3])
                data['throttle'][2, j] = float(a[4])
                data['speed'][2, j] = float(a[6]) * 1.609344 / 3.6  # to m/s
                data['dt'][2, j] = dt

            j += 1
    print('')

    # Compute trajectory.
    data['x'] = np.zeros((nb_types, nb_imgs, 2), dtype=np.float32)
    data['alpha'] = np.zeros((nb_types, nb_imgs), dtype=np.float32)
    data['x'][0], data['alpha'][0] = trajectory(data['dt'][0],
                                                data['speed'][0],
                                                data['angle'][0])
    # Left - right copy of x and alpha.
    if nb_types > 1:
        data['x'][2] = data['x'][1] = data['x'][0]
        data['alpha'][2] = data['alpha'][1] = data['alpha'][0]

    # Post-processing: curvature angle.
    cv_scales = [2, 3, 4, 6, 8]
    for s in cv_scales:
        k = 'angle_cv%i' % s
        data[k] = np.zeros((nb_types, nb_imgs), dtype=np.float32)
        data[k][0] = angle_curvature(data['x'][0], delta=s)
    if nb_types > 1:
        for s in cv_scales:
            k = 'angle_cv%i' % s
            data[k][2] = data[k][1] = data[k][0]

    # Post-processing: post-angles.
    post_scales = [5, 10, 20, 30, 40, 50]
    for s in post_scales:
        k = 'angle_post%i' % s
        data[k] = np.zeros((nb_types, nb_imgs), dtype=np.float32)
        data[k][0] = angle_post(data['alpha'][0],
                                data['dt'][0], data['speed'][0],
                                delta=s, offset=0.0)
    if nb_types > 1:
        for s in post_scales:
            k = 'angle_post%i' % s
            data[k][1] = angle_post(data['alpha'][0],
                                    data['dt'][0], data['speed'][0],
                                    delta=s,
                                    offset=-CAR_OFFSET)
            data[k][2] = angle_post(data['alpha'][0],
                                    data['dt'][0], data['speed'][0],
                                    delta=s,
                                    offset=CAR_OFFSET)

    med_scales = [5, 10, 15, 20]
    for s in med_scales:
        k = 'angle_med%i' % s
        data[k] = np.zeros((nb_types, nb_imgs), dtype=np.float32)
        data[k][0] = angle_median(data['alpha'][0],
                                  data['dt'][0], data['speed'][0],
                                  delta=s, offset=0.0)
    if nb_types > 1:
        for s in med_scales:
            k = 'angle_med%i' % s
            data[k][1] = angle_median(data['alpha'][0],
                                      data['dt'][0], data['speed'][0],
                                      delta=s,
                                      offset=-CAR_OFFSET)
            data[k][2] = angle_median(data['alpha'][0],
                                      data['dt'][0], data['speed'][0],
                                      delta=s,
                                      offset=CAR_OFFSET)

    # Cutting last values.
    cutting = 30
    for k in data.keys():
        data[k] = data[k][:, :-cutting]
    # Reshape elements.
    for k in data.keys():
        shape = data[k].shape
        new_shape = (shape[0] * shape[1], *shape[2:])
        data[k] = np.reshape(data[k], new_shape)

    # Mask data: keep frames after turning event only (1 frame ~ 0.1 second).
    if fmask is not None:
        mask = fmask(data['angle'])
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
    path = './data/5/'
    path = './data/q3_clean2/'
    path = './data/q3_recover_left2/'
    path = './data/q3_recover_right/'
    print('Dataset path: ', path)

    # Load data and 'pickle' dump.
    data = load_data(path, fmask=mask_negative)
    save_np_data(path, data)
    # dump_data(path, data)

    # HDF5 dataset.
    # create_hdf5(path)

if __name__ == '__main__':
    main()
