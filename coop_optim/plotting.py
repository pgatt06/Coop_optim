import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .kernel_utils import predict_from_alpha

font = {'family': 'sans', 'size': 12}
matplotlib.rc('font', **font)


def save_history_plot(histories, path, ylabel='Optimality gap', title=None):
    plt.figure(figsize=(6.5, 4.5))
    for label, values in histories.items():
        values = np.asarray(values, dtype=float)
        x = np.arange(1, len(values) + 1)
        plt.loglog(x, values, label=label)
    plt.grid(True, which='both')
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


def save_prediction_plot(x, y, x_grid, curves, path, title=None):
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(x, y, 'o', markersize=4, label='Data')
    for label, (x_m, alpha) in curves.items():
        y_hat = predict_from_alpha(x_grid, x_m, alpha)
        plt.plot(x_grid, y_hat, label=label)
    plt.grid(True)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
