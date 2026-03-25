import os
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import ensure_dir

plt.rc("font", family="sans-serif", size=12)


def save_history_plot(histories, path, ylabel, title, xlabel="Iteration"):
    plt.figure(figsize=(6.5, 4.5))
    for label, values in histories.items():
        values = np.asarray(values, dtype=float)
        x = np.arange(1, len(values) + 1)
        plt.loglog(x, np.maximum(values, 1e-16), label=label)
    plt.grid(True, which="both")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(path))
    plt.savefig(path)
    plt.close()


def save_semilogy_plot(histories, path, ylabel, title, xlabel="Communication rounds"):
    plt.figure(figsize=(6.5, 4.5))
    for label, values in histories.items():
        values = np.asarray(values, dtype=float)
        x = np.arange(len(values))
        plt.semilogy(x, np.maximum(values, 1e-16), label=label)
    plt.grid(True, which="both")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(path))
    plt.savefig(path)
    plt.close()


def save_agent_gap_grid_plot(histories, path, ylabel, title):
    n_panels = len(histories)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.8, 3.8 * nrows), squeeze=False)

    for ax, (label, gaps) in zip(axes.ravel(), histories.items()):
        gaps = np.asarray(gaps, dtype=float)
        x = np.arange(1, gaps.shape[0] + 1)
        for agent_id in range(gaps.shape[1]):
            ax.loglog(x, np.maximum(gaps[:, agent_id], 1e-16), lw=1.2, label=f"Agent {agent_id + 1}")
        ax.set_title(label)
        ax.grid(True, which="both")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)

    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)


def save_dataset_plot(x, y, path, landmark_indices):
    plt.figure(figsize=(6.5, 4.2))
    plt.scatter(x, y, s=18, alpha=0.75, label="Samples")
    plt.scatter(x[landmark_indices], y[landmark_indices], marker="*", s=110, color="crimson", label="Nyström landmarks")
    plt.grid(True, alpha=0.35)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Dataset and Nyström landmarks")
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(path))
    plt.savefig(path)
    plt.close()
