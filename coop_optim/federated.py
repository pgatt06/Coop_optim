import numpy as np

from .data_utils import as_1d
from .kernel_utils import cov_matrix, cross_cov_matrix


def build_federated_problem(X, Y, m=10, sigma=0.5, nu=1.0, landmarks=None):
    X = [as_1d(x_i) for x_i in X]
    Y = [as_1d(y_i) for y_i in Y]
    n_clients = len(X)

    if landmarks is None:
        x_m = np.linspace(-1.0, 1.0, m)
    else:
        x_m = as_1d(landmarks)
        m = x_m.size

    K_mm = cov_matrix(x_m)
    eye_m = np.eye(m)
    clients = []
    for client_id, (x_i, y_i) in enumerate(zip(X, Y)):
        K_i = cross_cov_matrix(x_i, x_m)
        H_i = K_i.T @ K_i + (sigma**2 / n_clients) * K_mm + (nu / n_clients) * eye_m
        b_i = K_i.T @ y_i
        eigvals = np.linalg.eigvalsh(H_i)
        clients.append(
            {
                "id": client_id,
                "x": x_i,
                "y": y_i,
                "K": K_i,
                "H": H_i,
                "b": b_i,
                "n": int(len(x_i)),
                "L": float(eigvals[-1]),
                "mu": float(eigvals[0]),
            }
        )

    H_total = np.zeros((m, m), dtype=float)
    b_total = np.zeros(m, dtype=float)
    for client in clients:
        H_total += client["H"]
        b_total += client["b"]
    alpha_star = np.linalg.solve(H_total, b_total)

    x_all = np.concatenate(X)
    y_all = np.concatenate(Y)
    K_all = cross_cov_matrix(x_all, x_m)

    def objective_fn(alpha):
        alpha = as_1d(alpha)
        residual = K_all @ alpha - y_all
        reg = (sigma**2) * alpha @ (K_mm @ alpha) + nu * (alpha @ alpha)
        return 0.5 * (residual @ residual) + 0.5 * reg

    return {
        "x_m": x_m,
        "K_mm": K_mm,
        "clients": clients,
        "alpha_star": alpha_star,
        "objective": objective_fn,
        "objective_star": float(objective_fn(alpha_star)),
        "n_clients": n_clients,
    }


def _batch_gradient(client, alpha, batch_idx, K_mm, sigma, nu, n_clients):
    alpha = as_1d(alpha)
    batch_idx = np.asarray(batch_idx, dtype=int)
    K_batch = client["K"][batch_idx]
    y_batch = client["y"][batch_idx]
    batch_size = len(batch_idx)
    if batch_size == 0:
        return np.zeros_like(alpha)
    grad_data = (client["n"] / batch_size) * (K_batch.T @ (K_batch @ alpha - y_batch))
    grad_reg = (sigma**2 / n_clients) * (K_mm @ alpha) + (nu / n_clients) * alpha
    return grad_data + grad_reg


def run_fedavg(
    clients,
    alpha_star,
    objective_fn,
    K_mm,
    sigma,
    nu,
    rounds=1200,
    B=20,
    C=5,
    E=1,
    lr0=1e-3,
    diminishing=False,
    seed=0,
):
    rng = np.random.default_rng(seed)
    n_clients = len(clients)
    alpha_global = np.zeros(K_mm.shape[0], dtype=float)
    objective_error = np.zeros(rounds + 1, dtype=float)
    F_star = float(objective_fn(alpha_star))
    objective_error[0] = float(objective_fn(alpha_global) - F_star)

    for round_id in range(rounds):
        eta = lr0 / np.sqrt(round_id + 1.0) if diminishing else lr0
        selected = rng.choice(n_clients, size=min(C, n_clients), replace=False)
        local_models = []
        for client_id in selected:
            client = clients[client_id]
            alpha_local = alpha_global.copy()
            indices = np.arange(client["n"])
            for _ in range(E):
                rng.shuffle(indices)
                for start in range(0, client["n"], B):
                    batch = indices[start : start + B]
                    grad = _batch_gradient(client, alpha_local, batch, K_mm, sigma, nu, n_clients)
                    alpha_local = alpha_local - eta * grad
            local_models.append(alpha_local)
        alpha_global = np.mean(local_models, axis=0)
        objective_error[round_id + 1] = float(objective_fn(alpha_global) - F_star)

    return alpha_global, objective_error


def run_scaffold(
    clients,
    alpha_star,
    objective_fn,
    K_mm,
    sigma,
    nu,
    rounds=1200,
    B=20,
    C=3,
    E=5,
    lr=1e-3,
    seed=0,
):
    rng = np.random.default_rng(seed)
    n_clients = len(clients)
    m = K_mm.shape[0]
    alpha = np.zeros(m, dtype=float)
    c_global = np.zeros(m, dtype=float)
    c_local = np.zeros((n_clients, m), dtype=float)

    objective_error = np.zeros(rounds + 1, dtype=float)
    F_star = float(objective_fn(alpha_star))
    objective_error[0] = float(objective_fn(alpha) - F_star)

    for round_id in range(rounds):
        selected = rng.choice(n_clients, size=min(C, n_clients), replace=False)
        delta_x = []
        delta_c_sum = np.zeros(m, dtype=float)

        for client_id in selected:
            client = clients[client_id]
            x_old = alpha.copy()
            x_new = x_old.copy()
            indices = np.arange(client["n"])
            steps = 0
            for _ in range(E):
                rng.shuffle(indices)
                for start in range(0, client["n"], B):
                    batch = indices[start : start + B]
                    grad = _batch_gradient(client, x_new, batch, K_mm, sigma, nu, n_clients)
                    x_new = x_new - lr * (grad - c_local[client_id] + c_global)
                    steps += 1

            delta_x.append(x_new - x_old)
            c_old = c_local[client_id].copy()
            c_new = c_old - c_global + (x_old - x_new) / max(1, steps) / lr
            c_local[client_id] = c_new
            delta_c_sum += c_new - c_old

        alpha = alpha + np.mean(delta_x, axis=0)
        c_global = c_global + delta_c_sum / float(n_clients)
        objective_error[round_id + 1] = float(objective_fn(alpha) - F_star)

    return alpha, objective_error
