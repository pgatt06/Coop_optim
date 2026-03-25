import argparse
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"


# Common constants
SIGMA = 0.5
NU = 1.0
SEED = 7

# Part I
N_PART1 = 100
M_PART1 = 10
N_AGENTS = 5
PART1_DGD_ITERS = 30000
PART1_GT_ITERS = 100000
PART1_DD_ITERS = 300000
PART1_ADMM_ITERS = 30000

# Part II
M_PART2 = 10
FEDAVG_REQUIRED_ROUNDS = {1: 25000, 5: 4000, 50: 1200}
FEDAVG_SWEEP_ROUNDS = 4000
FEDAVG_BATCH = 20
FEDAVG_SELECTED_CLIENTS = 5
FEDAVG_EPOCHS = (1, 5, 50)
SCAFFOLD_ROUNDS = 4000
SCAFFOLD_SELECTED_CLIENTS = 3

# Part III
DP_ITERS = 10000
DP_DELTA = 1e-5
DP_CLIP_NORM = 5.0
DP_EPSILONS = (0.1, 1.0, 10.0)


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
    X = [as_1d(x_i) for x_i in X]
    Y = [as_1d(y_i) for y_i in Y]
    return X, Y


def split_indices_equally(n, n_agents):
    if n % n_agents != 0:
        raise ValueError("n must be divisible by the number of agents.")
    block = n // n_agents
    return [np.arange(i * block, (i + 1) * block) for i in range(n_agents)]


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def rbf_kernel(x_data, x_landmarks):
    x = as_1d(x_data).reshape(-1, 1)
    z = as_1d(x_landmarks).reshape(1, -1)
    return np.exp(-((x - z) ** 2))


def cov_matrix(x_landmarks):
    return rbf_kernel(x_landmarks, x_landmarks)


def cross_cov_matrix(x_data, x_landmarks):
    return rbf_kernel(x_data, x_landmarks)


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


def solve_centralized(K_nm, y, K_mm, sigma=SIGMA, nu=NU):
    y = as_1d(y)
    m = K_mm.shape[0]
    H = K_nm.T @ K_nm + (sigma**2) * K_mm + nu * np.eye(m)
    b = K_nm.T @ y
    return np.linalg.solve(H, b)


def objective(alpha, K_nm, y, K_mm, sigma=SIGMA, nu=NU):
    alpha = as_1d(alpha)
    residual = K_nm @ alpha - as_1d(y)
    reg = (sigma**2) * alpha @ (K_mm @ alpha) + nu * (alpha @ alpha)
    return float(0.5 * (residual @ residual) + 0.5 * reg)


def smoothness_and_strong_convexity(K_nm, K_mm, sigma=SIGMA, nu=NU):
    H = K_nm.T @ K_nm + (sigma**2) * K_mm + nu * np.eye(K_mm.shape[0])
    eigvals = np.linalg.eigvalsh(H)
    return float(np.max(eigvals)), float(np.min(eigvals))


def quadratic_form_for_agent(K_i, y_i, K_mm, n_agents, sigma=SIGMA, nu=NU):
    m = K_mm.shape[0]
    H_i = K_i.T @ K_i + (sigma**2 / n_agents) * K_mm + (nu / n_agents) * np.eye(m)
    b_i = K_i.T @ as_1d(y_i)
    return H_i, b_i


def make_agent_data(problem, n_agents, sigma=SIGMA, nu=NU):
    K_nm = np.asarray(problem["K_nm"], dtype=float)
    y = as_1d(problem["y_n"])
    K_mm = np.asarray(problem["K_mm"], dtype=float)
    splits = split_indices_equally(len(y), n_agents)

    agents = []
    for agent_id, idx in enumerate(splits):
        K_i = K_nm[idx]
        y_i = y[idx]
        H_i, b_i = quadratic_form_for_agent(K_i, y_i, K_mm, n_agents, sigma=sigma, nu=nu)
        H_i = 0.5 * (H_i + H_i.T)
        eigvals = np.linalg.eigvalsh(H_i)
        if eigvals[0] <= 0:
            raise ValueError("Local Hessian must stay positive definite.")
        agents.append(
            {
                "id": agent_id,
                "indices": idx,
                "K": K_i,
                "y": y_i,
                "H": H_i,
                "b": b_i,
                "H_inv": np.linalg.inv(H_i),
                "L": float(eigvals[-1]),
                "mu": float(eigvals[0]),
                "n_local": int(len(idx)),
            }
        )
    return agents


def local_gradient(alpha, agent):
    alpha = as_1d(alpha)
    return agent["H"] @ alpha - agent["b"]


def grad_all(agents, alphas):
    alphas = np.asarray(alphas, dtype=float)
    return np.vstack([local_gradient(alphas[i], agents[i]) for i in range(len(agents))])


def optimality_gap(alphas, alpha_star):
    alphas = np.asarray(alphas, dtype=float)
    alpha_star = as_1d(alpha_star)
    return np.linalg.norm(alphas - alpha_star.reshape(1, -1), axis=1)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def adjacency_from_weights(W):
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")
    adj = (W > 0).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def is_connected(adj):
    adj = np.asarray(adj, dtype=int)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(adj[node]):
            if neighbor not in seen:
                seen.add(int(neighbor))
                stack.append(int(neighbor))
    return len(seen) == adj.shape[0]


def make_cycle_adjacency(n):
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i, (i - 1) % n] = 1
        adj[i, (i + 1) % n] = 1
    return adj


def make_line_adjacency(n):
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def make_complete_adjacency(n):
    adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    return adj


def make_small_world_adjacency(n, k=1, p=0.45, seed=0):
    if n <= 2:
        return make_complete_adjacency(n)

    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            adj[i, j] = 1
            adj[j, i] = 1

    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            if i >= j or rng.random() >= p:
                continue
            adj[i, j] = 0
            adj[j, i] = 0
            candidates = [u for u in range(n) if u != i and adj[i, u] == 0]
            if not candidates:
                adj[i, j] = 1
                adj[j, i] = 1
                continue
            new_j = int(rng.choice(candidates))
            adj[i, new_j] = 1
            adj[new_j, i] = 1

    if not is_connected(adj):
        return make_small_world_adjacency(n, k=k, p=p, seed=seed + 1)
    return adj


def metropolis_weights(adj):
    adj = np.asarray(adj, dtype=int)
    degrees = adj.sum(axis=1)
    n = adj.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.flatnonzero(adj[i]):
            W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def spectral_beta(W):
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    J = np.ones((n, n), dtype=float) / float(n)
    eigvals = np.linalg.eigvals(W - J)
    return float(np.max(np.abs(eigvals)))


def undirected_edges(adj):
    adj = np.asarray(adj, dtype=int)
    return [(i, j) for i in range(adj.shape[0]) for j in range(i + 1, adj.shape[1]) if adj[i, j] == 1]


def incidence_matrix(adj):
    edges = undirected_edges(adj)
    B = np.zeros((adj.shape[0], len(edges)), dtype=float)
    for edge_id, (i, j) in enumerate(edges):
        B[i, edge_id] = 1.0
        B[j, edge_id] = -1.0
    return B, edges


def get_graph(name, n, seed=0):
    if name == "cycle":
        adj = make_cycle_adjacency(n)
    elif name == "line":
        adj = make_line_adjacency(n)
    elif name in ("small_world", "small-world"):
        adj = make_small_world_adjacency(n, seed=seed)
    elif name in ("complete", "full", "fully_connected", "fully-connected"):
        adj = make_complete_adjacency(n)
    else:
        raise ValueError(f"Unknown graph: {name}")
    return adj, metropolis_weights(adj)


# ---------------------------------------------------------------------------
# Distributed optimization
# ---------------------------------------------------------------------------


def _init_alphas(n_agents, m, x0=None, seed=0, random_init=False):
    if x0 is not None:
        x0 = as_1d(x0)
        if x0.size != m:
            raise ValueError("x0 must have size m.")
        return np.tile(x0, (n_agents, 1))
    if random_init:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1e-3, size=(n_agents, m))
    return np.zeros((n_agents, m), dtype=float)


def _init_history():
    return {"agent_gaps": [], "bar_gap": [], "mean_gap": [], "max_gap": [], "consensus_gap": []}


def _record_history(history, alphas, alpha_star):
    gaps = optimality_gap(alphas, alpha_star)
    alpha_bar = np.mean(alphas, axis=0, keepdims=True)
    history["agent_gaps"].append(gaps.copy())
    history["mean_gap"].append(float(np.mean(gaps)))
    history["max_gap"].append(float(np.max(gaps)))
    history["bar_gap"].append(float(np.linalg.norm(alpha_bar.reshape(-1) - as_1d(alpha_star))))
    history["consensus_gap"].append(float(np.max(np.linalg.norm(alphas - alpha_bar, axis=1))))


def _finalize_history(history, alphas):
    for key in ("agent_gaps", "bar_gap", "mean_gap", "max_gap", "consensus_gap"):
        history[key] = np.asarray(history[key], dtype=float)
    history["alphas"] = alphas
    history["alpha_mean"] = np.mean(alphas, axis=0)
    return history


def _as_adjacency(graph):
    graph = np.asarray(graph)
    if graph.shape[0] == graph.shape[1] and np.all(np.isin(graph, [0, 1])):
        return graph.astype(int)
    if graph.shape[0] == graph.shape[1]:
        return adjacency_from_weights(graph)
    raise ValueError("Expected an adjacency or weight matrix.")


def _as_incidence(graph):
    graph = np.asarray(graph, dtype=float)
    if graph.shape[0] == graph.shape[1]:
        return incidence_matrix(_as_adjacency(graph))[0]
    return graph


def _block_diag(blocks):
    total = sum(block.shape[0] for block in blocks)
    out = np.zeros((total, total), dtype=float)
    start = 0
    for block in blocks:
        size = block.shape[0]
        out[start : start + size, start : start + size] = block
        start += size
    return out


def dual_lipschitz_constant(agents, graph):
    B = _as_incidence(graph)
    m = agents[0]["H"].shape[0]
    H_inv = _block_diag([agent["H_inv"] for agent in agents])
    B_big = np.kron(B, np.eye(m))
    eigvals = np.linalg.eigvalsh(B_big.T @ H_inv @ B_big)
    return float(max(np.max(eigvals), 1e-12))


def run_dgd(agents, W, alpha_star, step, n_iters, x0=None, seed=0, random_init=False):
    W = np.asarray(W, dtype=float)
    alphas = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
    history = _init_history()
    for _ in range(n_iters):
        grads = grad_all(agents, alphas)
        alphas = W @ alphas - step * grads
        _record_history(history, alphas, alpha_star)
    return _finalize_history(history, alphas)


def run_gradient_tracking(agents, W, alpha_star, step, n_iters, x0=None, seed=0, random_init=False):
    W = np.asarray(W, dtype=float)
    alphas = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
    grads = grad_all(agents, alphas)
    trackers = grads.copy()
    history = _init_history()
    for _ in range(n_iters):
        alphas_next = W @ alphas - step * trackers
        grads_next = grad_all(agents, alphas_next)
        trackers = W @ trackers + (grads_next - grads)
        alphas = alphas_next
        grads = grads_next
        _record_history(history, alphas, alpha_star)
    return _finalize_history(history, alphas)


def run_dual_decomposition(agents, graph, alpha_star, step, n_iters):
    B = _as_incidence(graph)
    n_edges = B.shape[1]
    m = alpha_star.size
    alphas = np.zeros((len(agents), m), dtype=float)
    lambdas = np.zeros((n_edges, m), dtype=float)
    history = _init_history()
    for _ in range(n_iters):
        dual_terms = B @ lambdas
        for i, agent in enumerate(agents):
            alphas[i] = agent["H_inv"] @ (agent["b"] - dual_terms[i])
        lambdas = lambdas + step * (B.T @ alphas)
        _record_history(history, alphas, alpha_star)
    history = _finalize_history(history, alphas)
    history["lambdas"] = lambdas
    return history


def run_consensus_admm(agents, graph, alpha_star, rho, n_iters):
    adj = _as_adjacency(graph)
    edges = undirected_edges(adj)
    n_agents = len(agents)
    m = alpha_star.size

    alphas = np.zeros((n_agents, m), dtype=float)
    y_edges = np.zeros((len(edges), m), dtype=float)
    lambda_left = np.zeros((len(edges), m), dtype=float)
    lambda_right = np.zeros((len(edges), m), dtype=float)

    incident_edges = [[] for _ in range(n_agents)]
    for edge_id, (i, j) in enumerate(edges):
        incident_edges[i].append((edge_id, 0))
        incident_edges[j].append((edge_id, 1))

    solvers = []
    for i, agent in enumerate(agents):
        degree_i = len(incident_edges[i])
        solvers.append(np.linalg.inv(agent["H"] + rho * degree_i * np.eye(m)))

    history = _init_history()
    for _ in range(n_iters):
        new_alphas = np.zeros_like(alphas)
        for i, agent in enumerate(agents):
            rhs = agent["b"].copy()
            for edge_id, side in incident_edges[i]:
                if side == 0:
                    rhs += rho * y_edges[edge_id] - lambda_left[edge_id]
                else:
                    rhs += rho * y_edges[edge_id] - lambda_right[edge_id]
            new_alphas[i] = solvers[i] @ rhs

        alphas = new_alphas
        for edge_id, (i, j) in enumerate(edges):
            y_edges[edge_id] = 0.5 * (
                alphas[i] + lambda_left[edge_id] / rho + alphas[j] + lambda_right[edge_id] / rho
            )
        for edge_id, (i, j) in enumerate(edges):
            lambda_left[edge_id] += rho * (alphas[i] - y_edges[edge_id])
            lambda_right[edge_id] += rho * (alphas[j] - y_edges[edge_id])
        _record_history(history, alphas, alpha_star)

    history = _finalize_history(history, alphas)
    history["y_edges"] = y_edges
    history["lambda_left"] = lambda_left
    history["lambda_right"] = lambda_right
    return history


def clip_rows(gradients, clip_norm):
    gradients = np.asarray(gradients, dtype=float)
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / (norms + 1e-12))
    return gradients * scale


def run_dgd_dp(
    agents,
    W,
    alpha_star,
    step,
    epsilon,
    n_iters,
    delta=1e-5,
    clip_norm=5.0,
    x0=None,
    seed=0,
    random_init=False,
):
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=float)
    alphas = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
    sensitivity = 2.0 * clip_norm
    noise_std = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / max(float(epsilon), 1e-12)
    noise_std /= np.sqrt(max(1, n_iters))

    history = _init_history()
    for _ in range(n_iters):
        grads = clip_rows(grad_all(agents, alphas), clip_norm=clip_norm)
        noise = rng.normal(0.0, noise_std, size=grads.shape)
        alphas = W @ alphas - step * (grads + noise)
        _record_history(history, alphas, alpha_star)

    history = _finalize_history(history, alphas)
    history["noise_std"] = float(noise_std)
    return history


# ---------------------------------------------------------------------------
# Federated learning
# ---------------------------------------------------------------------------


def build_federated_problem(X, Y, m=10, sigma=SIGMA, nu=NU, landmarks=None):
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------


GRAPH_SPECS = [
    ("cycle", "cycle"),
    ("line", "line"),
    ("small_world", "small-world"),
    ("complete", "complete"),
]


def part1_steps(agents, W, incidence):
    L_max = max(agent["L"] for agent in agents)
    mu_min = min(agent["mu"] for agent in agents)
    beta = spectral_beta(W)
    L_dual = dual_lipschitz_constant(agents, incidence)
    return {
        "L_max": float(L_max),
        "mu_min": float(mu_min),
        "beta": float(beta),
        "L_dual": float(L_dual),
        "eta_dgd": float(min(0.9 * (1.0 - beta) / L_max, 0.9 / L_max)),
        "eta_gt": float(0.2 / L_max),
        "tau_dual": float(1.0 / (2.0 * L_dual)),
        "rho_admm": float(np.sqrt(mu_min * L_max)),
    }


def run_part1():
    ensure_dir(str(FIGURES_DIR))
    first_db = DATA_DIR / "first_database.pkl"
    if not first_db.exists():
        raise FileNotFoundError("Missing data/first_database.pkl")

    x, y = load_first_database(first_db)
    problem = build_nystrom_problem(x, y, n=N_PART1, m=M_PART1, selection=True, seed=SEED)
    alpha_star = solve_centralized(problem["K_nm"], problem["y_n"], problem["K_mm"], sigma=SIGMA, nu=NU)
    obj_star = objective(alpha_star, problem["K_nm"], problem["y_n"], problem["K_mm"], sigma=SIGMA, nu=NU)
    L_central, mu_central = smoothness_and_strong_convexity(problem["K_nm"], problem["K_mm"], sigma=SIGMA, nu=NU)
    agents = make_agent_data(problem, N_AGENTS, sigma=SIGMA, nu=NU)

    local_sizes = [agent["n_local"] for agent in agents]
    if local_sizes != [N_PART1 // N_AGENTS] * N_AGENTS:
        raise ValueError("The 100 points must be split evenly across the 5 agents.")

    print("Part I")
    print(f"  n = {N_PART1}, m = {M_PART1}, agents = {N_AGENTS}")
    print(f"  centralized objective = {obj_star:.6f}")
    print(f"  centralized L = {L_central:.6f}")
    print(f"  centralized mu = {mu_central:.6f}")

    save_dataset_plot(
        problem["x_n"],
        problem["y_n"],
        FIGURES_DIR / "part1_dataset.pdf",
        landmark_indices=problem["landmark_indices"],
    )

    adj_line, W_line = get_graph("line", N_AGENTS, seed=SEED)
    incidence_line, _ = incidence_matrix(adj_line)
    line_steps = part1_steps(agents, W_line, incidence_line)

    hist_line_dgd = run_dgd(agents, W_line, alpha_star, step=line_steps["eta_dgd"], n_iters=PART1_DGD_ITERS, seed=SEED)
    hist_line_gt = run_gradient_tracking(
        agents, W_line, alpha_star, step=line_steps["eta_gt"], n_iters=PART1_GT_ITERS, seed=SEED
    )
    hist_line_dd = run_dual_decomposition(
        agents, incidence_line, alpha_star, step=line_steps["tau_dual"], n_iters=PART1_DD_ITERS
    )
    hist_line_admm = run_consensus_admm(
        agents, adj_line, alpha_star, rho=line_steps["rho_admm"], n_iters=PART1_ADMM_ITERS
    )

    save_agent_gap_grid_plot(
        {
            "DGD": hist_line_dgd["agent_gaps"],
            "Gradient tracking": hist_line_gt["agent_gaps"],
            "Dual decomposition": hist_line_dd["agent_gaps"],
            "ADMM": hist_line_admm["agent_gaps"],
        },
        FIGURES_DIR / "part1_gap_line.pdf",
        ylabel=r"$\|\alpha_i^t-\alpha^\star\|$",
        title="Part I - line graph",
    )

    dgd_by_graph = {}
    gt_by_graph = {}
    for graph_name, graph_label in GRAPH_SPECS:
        adj, W = get_graph(graph_name, N_AGENTS, seed=SEED)
        incidence, _ = incidence_matrix(adj)
        params = part1_steps(agents, W, incidence)
        hist_dgd = run_dgd(agents, W, alpha_star, step=params["eta_dgd"], n_iters=PART1_DGD_ITERS, seed=SEED)
        hist_gt = run_gradient_tracking(agents, W, alpha_star, step=params["eta_gt"], n_iters=PART1_GT_ITERS, seed=SEED)
        dgd_by_graph[graph_label] = hist_dgd["bar_gap"]
        gt_by_graph[graph_label] = hist_gt["bar_gap"]
        print(f"  [{graph_label}] beta = {params['beta']:.6f}, eta_dgd = {params['eta_dgd']:.3e}, eta_gt = {params['eta_gt']:.3e}")

    print(f"  line graph tau_dual = {line_steps['tau_dual']:.3e}")
    print(f"  line graph rho_admm = {line_steps['rho_admm']:.3e}")
    print(f"  final DGD bar gap = {hist_line_dgd['bar_gap'][-1]:.6e}")
    print(f"  final GT bar gap = {hist_line_gt['bar_gap'][-1]:.6e}")
    print(f"  final dual bar gap = {hist_line_dd['bar_gap'][-1]:.6e}")
    print(f"  final ADMM bar gap = {hist_line_admm['bar_gap'][-1]:.6e}")

    save_history_plot(
        dgd_by_graph,
        FIGURES_DIR / "part1_dgd_graph_compare.pdf",
        ylabel=r"$\|\bar{\alpha}^t-\alpha^\star\|$",
        title="DGD - graph effect on the averaged iterate",
    )
    save_history_plot(
        gt_by_graph,
        FIGURES_DIR / "part1_gt_graph_compare.pdf",
        ylabel=r"$\|\bar{\alpha}^t-\alpha^\star\|$",
        title="Gradient tracking - graph effect on the averaged iterate",
    )


# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------


def run_part2():
    ensure_dir(str(FIGURES_DIR))
    second_db = DATA_DIR / "second_database.pkl"
    if not second_db.exists():
        raise FileNotFoundError("Missing data/second_database.pkl")

    X, Y = load_second_database(second_db)
    fed_problem = build_federated_problem(X, Y, m=M_PART2, sigma=SIGMA, nu=NU)
    clients = fed_problem["clients"]
    client_sizes = [client["n"] for client in clients]
    if client_sizes != [20] * len(clients):
        raise ValueError("This setup expects 5 clients with 20 samples each.")

    L_max = max(client["L"] for client in clients)
    lr_const = 0.25 / L_max

    print("Part II")
    print(f"  clients = {len(clients)}, local sizes = {client_sizes}")
    print(f"  centralized objective = {fed_problem['objective_star']:.6f}")
    print(f"  local L_max = {L_max:.6f}")
    print(f"  FedAvg step = {lr_const:.6e}")

    fedavg_curves = {}
    for E in FEDAVG_EPOCHS:
        lr = lr_const if E < 50 else 0.5 * lr_const
        rounds = FEDAVG_REQUIRED_ROUNDS[E]
        _, curve = run_fedavg(
            clients=clients,
            alpha_star=fed_problem["alpha_star"],
            objective_fn=fed_problem["objective"],
            K_mm=fed_problem["K_mm"],
            sigma=SIGMA,
            nu=NU,
            rounds=rounds,
            B=FEDAVG_BATCH,
            C=FEDAVG_SELECTED_CLIENTS,
            E=E,
            lr0=lr,
            diminishing=False,
            seed=SEED,
        )
        fedavg_curves[f"E={E}"] = curve
        print(f"  FedAvg E={E}: final objective error = {curve[-1]:.6e}")

    save_semilogy_plot(
        fedavg_curves,
        FIGURES_DIR / "part2_fedavg_required_E.pdf",
        ylabel="Objective error",
        title="FedAvg - B=20, C=5",
    )

    sweep_curves = {}
    sweep_settings = [
        ("B=20, C=5, E=5, constant step", dict(B=20, C=5, E=5, diminishing=False, lr0=lr_const)),
        ("B=10, C=5, E=5, constant step", dict(B=10, C=5, E=5, diminishing=False, lr0=lr_const)),
        ("B=5, C=3, E=5, constant step", dict(B=5, C=3, E=5, diminishing=False, lr0=lr_const)),
        ("B=20, C=5, E=50, decreasing step", dict(B=20, C=5, E=50, diminishing=True, lr0=lr_const)),
    ]
    for label, params in sweep_settings:
        _, curve = run_fedavg(
            clients=clients,
            alpha_star=fed_problem["alpha_star"],
            objective_fn=fed_problem["objective"],
            K_mm=fed_problem["K_mm"],
            sigma=SIGMA,
            nu=NU,
            rounds=FEDAVG_SWEEP_ROUNDS,
            seed=SEED + 1,
            **params,
        )
        sweep_curves[label] = curve

    save_semilogy_plot(
        sweep_curves,
        FIGURES_DIR / "part2_fedavg_param_sweep.pdf",
        ylabel="Objective error",
        title="FedAvg - effect of B, C, E, and the step schedule",
    )

    _, fedavg_curve = run_fedavg(
        clients=clients,
        alpha_star=fed_problem["alpha_star"],
        objective_fn=fed_problem["objective"],
        K_mm=fed_problem["K_mm"],
        sigma=SIGMA,
        nu=NU,
        rounds=SCAFFOLD_ROUNDS,
        B=FEDAVG_BATCH,
        C=SCAFFOLD_SELECTED_CLIENTS,
        E=5,
        lr0=lr_const,
        diminishing=False,
        seed=SEED + 2,
    )
    _, scaffold_curve = run_scaffold(
        clients=clients,
        alpha_star=fed_problem["alpha_star"],
        objective_fn=fed_problem["objective"],
        K_mm=fed_problem["K_mm"],
        sigma=SIGMA,
        nu=NU,
        rounds=SCAFFOLD_ROUNDS,
        B=FEDAVG_BATCH,
        C=SCAFFOLD_SELECTED_CLIENTS,
        E=5,
        lr=lr_const,
        seed=SEED + 2,
    )
    print(f"  optional FedAvg (C=3, E=5): final objective error = {fedavg_curve[-1]:.6e}")
    print(f"  optional SCAFFOLD: final objective error = {scaffold_curve[-1]:.6e}")


# ---------------------------------------------------------------------------
# Part III
# ---------------------------------------------------------------------------


def run_part3():
    ensure_dir(str(FIGURES_DIR))
    first_db = DATA_DIR / "first_database.pkl"
    if not first_db.exists():
        raise FileNotFoundError("Missing data/first_database.pkl")

    x, y = load_first_database(first_db)
    problem = build_nystrom_problem(x, y, n=N_PART1, m=M_PART1, selection=True, seed=SEED)
    alpha_star = solve_centralized(problem["K_nm"], problem["y_n"], problem["K_mm"], sigma=SIGMA, nu=NU)
    agents = make_agent_data(problem, N_AGENTS, sigma=SIGMA, nu=NU)

    _, W_line = get_graph("line", N_AGENTS, seed=SEED)
    L_max = max(agent["L"] for agent in agents)
    beta = spectral_beta(W_line)
    eta_dgd = min(0.9 * (1.0 - beta) / L_max, 0.9 / L_max)

    print("Part III")
    print(f"  line-graph beta = {beta:.6f}")
    print(f"  DGD step = {eta_dgd:.6e}")

    baseline = run_dgd(agents, W_line, alpha_star, step=eta_dgd, n_iters=DP_ITERS, seed=SEED)
    histories = {"Non-private DGD": baseline["agent_gaps"]}
    print(f"  non-private final bar gap = {baseline['bar_gap'][-1]:.6e}")

    for epsilon in DP_EPSILONS:
        hist = run_dgd_dp(
            agents,
            W_line,
            alpha_star,
            step=eta_dgd,
            epsilon=epsilon,
            n_iters=DP_ITERS,
            delta=DP_DELTA,
            clip_norm=DP_CLIP_NORM,
            seed=SEED,
        )
        histories[f"epsilon={epsilon}"] = hist["agent_gaps"]
        print(f"  epsilon={epsilon}: noise_std = {hist['noise_std']:.6e}, final bar gap = {hist['bar_gap'][-1]:.6e}")

    save_agent_gap_grid_plot(
        histories,
        FIGURES_DIR / "part3_dgd_dp_epsilons.pdf",
        ylabel=r"$\|\alpha_i^t-\alpha^\star\|$",
        title="Part III - private DGD",
    )


def run_all_parts():
    print("\n===== Part I =====")
    run_part1()
    print("\n===== Part II =====")
    run_part2()
    print("\n===== Part III =====")
    run_part3()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Cooperative Kernel Regression project.")
    parser.add_argument(
        "--part",
        choices=("1", "2", "3", "all"),
        default="all",
        help="Choose which part to run. Default: all.",
    )
    args = parser.parse_args(argv)

    if args.part == "1":
        print("\n===== Part I =====")
        run_part1()
    elif args.part == "2":
        print("\n===== Part II =====")
        run_part2()
    elif args.part == "3":
        print("\n===== Part III =====")
        run_part3()
    else:
        run_all_parts()


if __name__ == "__main__":
    main()
