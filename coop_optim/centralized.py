import numpy as np

from .data_utils import as_1d
from .kernel_utils import cov_matrix, cross_cov_matrix


def build_nystrom_problem(x, y, n=100, m=10, selection=True, seed=0, landmarks=None):
    x = as_1d(x)
    y = as_1d(y)
    x_n = x[:n]
    y_n = y[:n]

    if landmarks is not None:
        x_m = as_1d(landmarks)
        landmark_indices = np.array([], dtype=int)
    else:
        rng = np.random.default_rng(seed)
        if selection:
            landmark_indices = np.sort(rng.choice(np.arange(n), size=m, replace=False))
            x_m = x_n[landmark_indices]
        else:
            landmark_indices = np.array([], dtype=int)
            x_m = np.linspace(-1.0, 1.0, m)

    K_nm = cross_cov_matrix(x_n, x_m)
    K_mm = cov_matrix(x_m)
    return {
        "x_n": x_n,
        "y_n": y_n,
        "x_m": x_m,
        "K_nm": K_nm,
        "K_mm": K_mm,
        "landmark_indices": landmark_indices,
    }


def solve_centralized(K_nm, y, K_mm, sigma=0.5, nu=1.0):
    y = as_1d(y)
    m = K_mm.shape[0]
    H = K_nm.T @ K_nm + (sigma**2) * K_mm + nu * np.eye(m)
    b = K_nm.T @ y
    return np.linalg.solve(H, b)


def objective(alpha, K_nm, y, K_mm, sigma=0.5, nu=1.0):
    alpha = as_1d(alpha)
    residual = K_nm @ alpha - as_1d(y)
    reg = (sigma**2) * alpha @ (K_mm @ alpha) + nu * (alpha @ alpha)
    return float(0.5 * (residual @ residual) + 0.5 * reg)


def smoothness_and_strong_convexity(K_nm, K_mm, sigma=0.5, nu=1.0):
    H = K_nm.T @ K_nm + (sigma**2) * K_mm + nu * np.eye(K_mm.shape[0])
    eigvals = np.linalg.eigvalsh(H)
    return float(np.max(eigvals)), float(np.min(eigvals))


def quadratic_form_for_agent(K_i, y_i, K_mm, n_agents, sigma=0.5, nu=1.0):
    m = K_mm.shape[0]
    H_i = K_i.T @ K_i + (sigma**2 / n_agents) * K_mm + (nu / n_agents) * np.eye(m)
    b_i = K_i.T @ as_1d(y_i)
    return H_i, b_i
