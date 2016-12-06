"""Random surfaces generation.
"""
import numpy as np

import numba


def hyperplane_rand(shape):
    """Random hyperplane in the 2D plane. Random uniform output, symmetric
    between the two planes.
    """
    # Random straight line.
    x1 = np.random.randint(0, shape[0])
    y1 = np.random.randint(0, shape[1])
    a = np.tan(np.random.rand(1) * np.pi)

    # Grid.
    yv, xv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    bsurf = (a * (xv - x1) + y1) <= yv

    # Random surface.
    rand_surf = bsurf.astype(np.float32) * 2 - 1
    rand_surf = rand_surf * np.random.uniform(-1., 1.)
    return rand_surf


@numba.jit(nopython=True, cache=True)
def fbm2d_midpoint(shape, H, stationary=False):
    """Simulation of a 2D fBm (somehow!) using the midpoint algorithm.

    This algorithm is directly inspired by the Levy construction of Brownian
    motion by constructing a piecewise linear approximation refined at every step.
    This construction does not converge stricto-sensus to a fBm, since the
    long range dependency is not corresponding to the fBm. Nevertheless, local
    stochastic structure (i.e. LND) is similar to the fBm.

    Original reference on the subject:
    http://excelsior.biosci.ohio-state.edu/~carlson/history/PDFs/p371-fournier.pdf

    Params:
      shape: 2D shape.
      H: Hurst exponent.
      stationary: If false, generate an approximation on the grid [0,1]x[0,1].x
      If True, generate an approximation with constant variance.

    Return:
      Approximation of 2D fBm.
    """
    # return np.zeros(shape, dtype=np.float32)

    # Find the number of levels!
    N = max(shape[0], shape[1])
    nlevels = int(np.ceil(np.log(N-1) / np.log(2)))

    # First grid approximation.
    # Z = np.zeros((2, 2), dtype=np.float32)
    if not stationary:
        Z = np.zeros((2, 2), dtype=np.float32)
        Z[0, 0] = 0.0
        Z[0, 1] = np.random.normal(0., 1.)
        Z[1, 0] = np.random.normal(0., 1.)
        Z[1, 1] = np.random.normal(0., 1.) * np.sqrt(2**H)
    else:
        # Z[0, 0] = np.random.normal(0., 1.)
        # Z[0, 1] = np.random.normal(0., 1.)
        # Z[1, 0] = np.random.normal(0., 1.)
        # Z[1, 1] = np.random.normal(0., 1.)
        Z = np.random.randn(2, 2).astype(np.float32)

    for lv in range(1, nlevels+1):
        Y = np.zeros((2**lv+1, 2**lv+1), dtype=np.float32)
        # Copy previous resolution.
        Y[::2, ::2] = Z

        # Variance of intermediate points.
        varc = (1. - 1./4*2**H - 1./8*2**(2*H)) * 2**(-2*lv*H+H)
        vara = (1. - 2**(2*H-2)) * 2**(-2*lv*H)

        # Linear interpolation.
        Y[1::2, ::2] += 0.5 * Z[1:, :] + 0.5 * Z[:-1, :]
        Y[::2, 1::2] += 0.5 * Z[:, 1:] + 0.5 * Z[:, :-1]

        Y[1::2, 1::2] += 0.25 * Z[1:, 1:] + 0.25 * Z[:-1, :-1] + \
            0.25 * Z[:-1, 1:] + 0.25 * Z[1:, :-1]

        # Add random noise.
        zshape = Z.shape
        Y[1::2, ::2] += np.random.randn(zshape[0]-1, zshape[1]) * np.sqrt(vara)
        Y[::2, 1::2] += np.random.randn(zshape[0], zshape[1]-1) * np.sqrt(vara)
        Y[1::2, 1::2] += np.random.randn(zshape[0]-1, zshape[1]-1) * np.sqrt(varc)

        # # Simulation of remaining intermediate points.
        # for m in range(2**(lv-1)):
        #     for l in range(2**(lv-1)):
        #         x = 2*l + 1
        #         y = 2*m + 1

        #         # Center - right - top points.
        #         Y[x, y] = np.sqrt(varc) * np.random.normal() + \
        #             0.25 * (Y[x-1, y-1] + Y[x-1, y+1] + Y[x+1, y+1] + Y[x+1, y-1])

        #         Y[x+1, y] = np.sqrt(vara) * np.random.normal() + \
        #             0.5 * (Y[x+1, y-1] + Y[x+1, y+1])

        #         Y[x, y+1] = np.sqrt(vara) * np.random.normal() + \
        #             0.5 * (Y[x-1, y+1] + Y[x+1, y+1])

        #         # Point below and left (left and right axis).
        #         if m == 0:
        #             Y[x, y-1] = np.sqrt(vara) * np.random.normal() + \
        #                 0.5 * (Y[x-1, y-1] + Y[x+1, y-1])
        #         if l == 0:
        #             Y[x-1, y] = np.sqrt(vara) * np.random.normal() + \
        #                 0.5 * (Y[x-1, y-1] + Y[x-1, y+1])

        Z = Y

    # Reshape.
    if not stationary:
        Z = Z[:shape[0], :shape[1]]
    else:
        x = (Z.shape[0] - shape[0]) // 2
        y = (Z.shape[1] - shape[1]) // 2
        Z = Z[x:x+shape[0], y:y+shape[1]]
    return Z


# @numba.jit(nopython=True, cache=True)
def surface_reflect(data, vmin, vmax):
    """Reflect a random surface on min and max values.
    """
    # while np.min(data) < vmin or np.max(data) > vmax:
    data = vmin + np.abs(data - vmin)
    data = vmax - np.abs(vmax - data)
    data = vmin + np.abs(data - vmin)
    data = vmax - np.abs(vmax - data)

    return data
