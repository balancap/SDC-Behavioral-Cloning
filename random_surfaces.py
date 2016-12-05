"""Random surfaces generation.
"""
import numpy as np

import numba


@numba.jit(nopython=True, cache=True)
def fbm2d_midpoint(H, nlevels, stationary=False):
    """Simulation of a 2D fBm (somehow!) using the midpoint algorithm.

    This algorithm is directly inspired by the Levy construction of Brownian
    motion by constructing a piecewise linear approximation refined at every step.
    This construction does not converge stricto-sensus to a fBm, since the
    long range dependency is not corresponding to the fBm. Nevertheless, local
    stochastic structure (i.e. LND) is similar to the fBm.

    Original reference on the subject:
    http://excelsior.biosci.ohio-state.edu/~carlson/history/PDFs/p371-fournier.pdf

    Params:
      H: Hurst exponent.
      nlevels: Number of levels in the construction.
      stationary: If false, generate an approximation on the grid [0,1]x[0,1].x
      If True, generate an approximation with constant variance.

    Return:
      2D fBm on a grid of size 2^nlevels + 1.
    """
    # First grid approximation.
    Z = np.zeros((2, 2), dtype=np.float32)
    if not stationary:
        Z[0, 0] = 0.0
        Z[0, 1] = np.random.normal(0., 1.)
        Z[1, 0] = np.random.normal(0., 1.)
        Z[1, 1] = np.random.normal(0., 1.) * np.sqrt(2**H)
    else:
        Z[0, 0] = np.random.normal(0., 1.)
        Z[0, 1] = np.random.normal(0., 1.)
        Z[1, 0] = np.random.normal(0., 1.)
        Z[1, 1] = np.random.normal(0., 1.)

    for lv in range(1, nlevels+1):
        Y = np.zeros((2**lv+1, 2**lv+1), dtype=np.float32)
        # Copy previous resolution.
        Y[::2, ::2] = Z

        # Variance of intermediate points.
        varc = (1. - 1./4*2**H - 1./8*2**(2*H)) * 2**(-2*lv*H+H)
        vara = (1. - 2**(2*H-2)) * 2**(-2*lv*H)

        # Simulation of remaining intermediate points.
        for m in range(2**(lv-1)):
            for l in range(2**(lv-1)):
                x = 2*l + 1
                y = 2*m + 1

                # Center - right - top points.
                Y[x, y] = np.sqrt(varc) * np.random.normal(1) + \
                    0.25 * (Y[x-1, y-1] + Y[x-1, y+1] + Y[x+1, y+1] + Y[x+1, y-1])

                Y[x+1, y] = np.sqrt(vara) * np.random.normal(1) + \
                    0.5 * (Y[x+1, y-1] + Y[x+1, y+1])

                Y[x, y+1] = np.sqrt(vara) * np.random.normal(1) + \
                    0.5 * (Y[x-1, y+1] + Y[x+1, y+1])

                # Point below and left (left and right axis).
                if m == 0:
                    Y[x, y-1] = np.sqrt(vara) * np.random.normal(1) + \
                        0.5 * (Y[x-1, y-1] + Y[x+1, y-1])
                if l == 0:
                    Y[x-1, y] = np.sqrt(vara) * np.random.normal(1) + \
                        0.5 * (Y[x-1, y-1] + Y[x-1, y+1])

        Z = Y
    return Z
