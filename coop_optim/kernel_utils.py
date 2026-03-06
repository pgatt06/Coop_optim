import numpy as np

import coop_optim.centralized_solution as cs


def as_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def cov_matrix(x_landmarks):
    return np.asarray(cs.Cov(as_1d(x_landmarks)), dtype=float)


def cross_cov_matrix(x_data, x_landmarks):
    return np.asarray(cs.Cov2(as_1d(x_data), as_1d(x_landmarks)), dtype=float)


def predict_from_alpha(x_query, x_landmarks, alpha):
    return cross_cov_matrix(x_query, x_landmarks) @ as_1d(alpha)
