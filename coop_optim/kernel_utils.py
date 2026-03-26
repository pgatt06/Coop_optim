import numpy as np

from .data_utils import as_1d


def rbf_kernel(x_data, x_landmarks):
    x = as_1d(x_data).reshape(-1, 1)
    z = as_1d(x_landmarks).reshape(1, -1)
    return np.exp(-((x - z) ** 2))


def cov_matrix(x_landmarks):
    return rbf_kernel(x_landmarks, x_landmarks)


def cross_cov_matrix(x_data, x_landmarks):
    return rbf_kernel(x_data, x_landmarks)


def predict(alpha, x_query, x_landmarks):
    return cross_cov_matrix(x_query, x_landmarks) @ as_1d(alpha)
