import numpy as np


def sphere(position):
    val = np.sum(np.square(position))
    return val
