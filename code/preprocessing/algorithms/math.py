import numpy as np


def log10_transform_data(data):
    minimum_log10_value = 0.001
    data[data < minimum_log10_value] = minimum_log10_value
    return np.log10(data)


def lorentzian_distance(x, y):
    return np.sum(np.log(1 + np.fabs(x - y)))


def angular_distance(x, y):
    xy = np.vstack([x, y])
    numerator = np.sum(np.prod(xy, axis=0))
    denominator = np.sqrt(np.sum(np.float_power(x, 2))) * np.sqrt(np.sum(np.float_power(y, 2)))
    return 1 - numerator/denominator
