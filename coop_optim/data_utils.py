import os
import pickle
import warnings

import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def as_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def load_first_database(path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"numpy\.core\.numeric is deprecated.*",
            category=DeprecationWarning,
        )
        with open(path, "rb") as handle:
            x, y = pickle.load(handle)
    return as_1d(x), as_1d(y)


def load_second_database(path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"numpy\.core\.numeric is deprecated.*",
            category=DeprecationWarning,
        )
        with open(path, "rb") as handle:
            X, Y = pickle.load(handle)
    return [as_1d(x_i) for x_i in X], [as_1d(y_i) for y_i in Y]


def split_indices_equally(n, n_agents):
    if n % n_agents != 0:
        raise ValueError("n must be divisible by the number of agents.")
    block = n // n_agents
    return [np.arange(i * block, (i + 1) * block) for i in range(n_agents)]
