""" Metrics """

import numpy as np

def ND(prediction, Y, mask=None):
    if mask is None:
        mask = np.array((~np.isnan(Y)).astype(int))
    Y[mask == 0] = 0.
    return abs((prediction - Y) * mask).sum() / abs(Y).sum()

def NRMSE(prediction, Y, mask=None):
    if mask is None:
        mask = np.array((~np.isnan(Y)).astype(int))
    Y[mask == 0] = 0.
    return pow((pow(prediction - Y, 2) * mask).sum(), 0.5) / abs(Y).sum() * pow(mask.sum(), 0.5)

