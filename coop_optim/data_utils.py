import os
import pickle
import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=float).reshape(-1)


def load_first_database(path):
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
    return _as_array(x), _as_array(y)


def load_second_database(path):
    with open(path, 'rb') as f:
        X, Y = pickle.load(f)
    X = [np.asarray(xi, dtype=float).reshape(-1) for xi in X]
    Y = [np.asarray(yi, dtype=float).reshape(-1) for yi in Y]
    return X, Y


def split_indices_equally(n, n_agents):
    if n % n_agents != 0:
        raise ValueError('n must be divisible by n_agents for this helper.')
    chunk = n // n_agents
    return [np.arange(i * chunk, (i + 1) * chunk) for i in range(n_agents)]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
