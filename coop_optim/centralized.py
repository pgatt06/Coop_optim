import numpy as np

import coop_optim.centralized_solution as cs


def build_nystrom_problem(x, y, n=100, m=10, selection=True, seed=0, landmarks=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if landmarks is not None:
        x_n = x[:n]
        y_n = y[:n]
        x_m = np.asarray(landmarks, dtype=float).reshape(-1)
        M = np.asarray(cs.Cov2(x_n, x_m), dtype=float)
        Kmm = np.asarray(cs.Cov(x_m), dtype=float)
        ind = np.array([], dtype=int)
    else:
        x_n, y_n, x_m, M, Kmm, ind = cs.build_nystrom_matrices(
            x, y, n=n, m=m, selection=selection, seed=seed
        )
        x_n = np.asarray(x_n, dtype=float)
        y_n = np.asarray(y_n, dtype=float)
        x_m = np.asarray(x_m, dtype=float)
        M = np.asarray(M, dtype=float)
        Kmm = np.asarray(Kmm, dtype=float)
        ind = np.asarray(ind, dtype=int)

    return {
        'x_n': x_n,
        'y_n': y_n,
        'x_m': x_m,
        'M': M,
        'Kmm': Kmm,
        'landmark_indices': ind,
    }


def solve_centralized(M, y, Kmm, sigma=0.5, nu=1.0):
    return np.asarray(cs.solve_from_matrices(M, y, Kmm, sigma=sigma, nu=nu), dtype=float)


def objective(alpha, M, y, Kmm, sigma=0.5, nu=1.0):
    return float(cs.objective(alpha, M, y, Kmm, sigma=sigma, nu=nu))


def gradient(alpha, M, y, Kmm, sigma=0.5, nu=1.0):
    return np.asarray(cs.gradient(alpha, M, y, Kmm, sigma=sigma, nu=nu), dtype=float)


def quadratic_form_for_agent(M_i, y_i, Kmm, n_agents, sigma=0.5, nu=1.0):
    M_i = np.asarray(M_i, dtype=float)
    y_i = np.asarray(y_i, dtype=float).reshape(-1)
    Kmm = np.asarray(Kmm, dtype=float)
    m = Kmm.shape[0]
    Q_i = M_i.T @ M_i + ((sigma ** 2) / n_agents) * Kmm + (nu / n_agents) * np.eye(m)
    b_i = M_i.T @ y_i
    return Q_i, b_i


def smoothness_and_strong_convexity(M, Kmm, sigma=0.5, nu=1.0):
    M = np.asarray(M, dtype=float)
    Kmm = np.asarray(Kmm, dtype=float)
    H = M.T @ M + (sigma ** 2) * Kmm + nu * np.eye(Kmm.shape[0])
    eigvals = np.linalg.eigvalsh(H)
    return float(np.max(eigvals)), float(np.min(eigvals))
