import argparse
import os
import pickle
import tempfile
import time
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
PART1_BREAK_ITERS = 2500
PART1_PACKET_LOSS = 0.35
PART1_ACTIVE_PROB = 0.35
PART1_SCALING_THRESHOLD = 1.0
PART1_SCALING_PROBE_ITERS = 120
PART1_SCALING_MAX_EVALS = 14
PART1_SCALING_BATCH_SIZE = 2048
PART1_SCALING_PER_EVAL_LIMIT_S = 90.0
PART1_SCALING_TOTAL_BUDGET_S = 900.0
PART1_SCALING_MIN_AGENTS = 5
PART1_SCALING_MAX_AGENTS = 100
PART1_SCALING_TARGET_LOCAL_SIZE = 20000
PART1_SCALING_LONG_ITER_BUDGET = 60000
PART1_SCALING_ETA_MULTIPLIERS = (0.2, 0.4, 0.6, 0.8)

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
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def as_1d(x):
    """Return x as a flat float array."""
    return np.asarray(x, dtype=float).reshape(-1)


def load_first_database(path):
    """Load the first database while silencing legacy pickle warnings."""
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
    """Load the second database and flatten each local dataset."""
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
    """Split n samples into equally sized contiguous blocks."""
    if n % n_agents != 0:
        raise ValueError("n must be divisible by the number of agents.")
    block = n // n_agents
    return [np.arange(i * block, (i + 1) * block) for i in range(n_agents)]


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def rbf_kernel(x_data, x_landmarks):
    """Evaluate the Gaussian kernel between data and landmark points."""
    x = as_1d(x_data).reshape(-1, 1)
    z = as_1d(x_landmarks).reshape(1, -1)
    return np.exp(-((x - z) ** 2))


def cov_matrix(x_landmarks):
    """Return the Gram matrix on the landmark set."""
    return rbf_kernel(x_landmarks, x_landmarks)


def cross_cov_matrix(x_data, x_landmarks):
    """Return the cross-kernel matrix between data and landmarks."""
    return rbf_kernel(x_data, x_landmarks)


def predict(alpha, x_query, x_landmarks):
    """Predict the reconstructed function on query points."""
    return cross_cov_matrix(x_query, x_landmarks) @ as_1d(alpha)


def build_nystrom_problem(x, y, n=100, m=10, selection=True, seed=0, landmarks=None):
    """Build the Nyström approximation used throughout the project."""
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
    """Solve the centralized kernel regression problem in closed form."""
    y = as_1d(y)
    m = K_mm.shape[0]
    H = K_nm.T @ K_nm + (sigma**2) * K_mm + nu * np.eye(m)
    b = K_nm.T @ y
    return np.linalg.solve(H, b)


def objective(alpha, K_nm, y, K_mm, sigma=SIGMA, nu=NU):
    """Evaluate the centralized objective value."""
    alpha = as_1d(alpha)
    residual = K_nm @ alpha - as_1d(y)
    reg = (sigma**2) * alpha @ (K_mm @ alpha) + nu * (alpha @ alpha)
    return float(0.5 * (residual @ residual) + 0.5 * reg)


def smoothness_and_strong_convexity(K_nm, K_mm, sigma=SIGMA, nu=NU):
    """Return the smoothness and strong convexity constants of the objective."""
    H = K_nm.T @ K_nm + (sigma**2) * K_mm + nu * np.eye(K_mm.shape[0])
    eigvals = np.linalg.eigvalsh(H)
    return float(np.max(eigvals)), float(np.min(eigvals))


def quadratic_form_for_agent(K_i, y_i, K_mm, n_agents, sigma=SIGMA, nu=NU):
    """Build the local quadratic model associated with one agent."""
    m = K_mm.shape[0]
    H_i = K_i.T @ K_i + (sigma**2 / n_agents) * K_mm + (nu / n_agents) * np.eye(m)
    b_i = K_i.T @ as_1d(y_i)
    return H_i, b_i


def make_agent_data(problem, n_agents, sigma=SIGMA, nu=NU):
    """Split the Nyström problem evenly and build one quadratic model per agent."""
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


def aggregate_quadratic_model(agents):
    """Sum the local quadratic models into the centralized one."""
    H_total = np.zeros_like(agents[0]["H"], dtype=float)
    b_total = np.zeros_like(agents[0]["b"], dtype=float)
    for agent in agents:
        H_total += np.asarray(agent["H"], dtype=float)
        b_total += np.asarray(agent["b"], dtype=float)
    return H_total, b_total


def centralized_solution_from_agents(agents):
    """Recover the centralized optimizer from local quadratic summaries."""
    H_total, b_total = aggregate_quadratic_model(agents)
    return np.linalg.solve(H_total, b_total)


def aggregate_gradient(alpha, agents):
    """Evaluate the centralized gradient using only local quadratic summaries."""
    H_total, b_total = aggregate_quadratic_model(agents)
    return H_total @ as_1d(alpha) - b_total


def aggregate_gradient_norm(alpha, agents):
    """Return a stationarity surrogate for large-n runs."""
    return float(np.linalg.norm(aggregate_gradient(alpha, agents)))


def make_streaming_agent_data(x_data, y_data, x_landmarks, n_agents, sigma=SIGMA, nu=NU, batch_size=2048):
    """Build local quadratic models in batches without storing the full Knm matrix."""
    x_data = as_1d(x_data)
    y_data = as_1d(y_data)
    x_landmarks = as_1d(x_landmarks)

    m = x_landmarks.size
    K_mm = cov_matrix(x_landmarks)
    eye_m = np.eye(m)
    splits = np.array_split(np.arange(len(y_data)), n_agents)

    agents = []
    for agent_id, idx in enumerate(splits):
        H_i = (sigma**2 / n_agents) * K_mm + (nu / n_agents) * eye_m
        b_i = np.zeros(m, dtype=float)

        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            K_batch = cross_cov_matrix(x_data[batch_idx], x_landmarks)
            y_batch = y_data[batch_idx]
            H_i += K_batch.T @ K_batch
            b_i += K_batch.T @ y_batch

        H_i = 0.5 * (H_i + H_i.T)
        eigvals = np.linalg.eigvalsh(H_i)
        if eigvals[0] <= 0:
            raise ValueError("Local Hessian must stay positive definite.")
        agents.append(
            {
                "id": agent_id,
                "indices": idx,
                "H": H_i,
                "b": b_i,
                "H_inv": np.linalg.inv(H_i),
                "L": float(eigvals[-1]),
                "mu": float(eigvals[0]),
                "n_local": int(len(idx)),
            }
        )
    return agents


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def adjacency_from_weights(W):
    """Convert a mixing matrix into an unweighted adjacency support."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")
    adj = (W > 0).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def is_connected(adj):
    """Return True when the undirected graph is connected."""
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
    """Build the undirected cycle graph on n nodes."""
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i, (i - 1) % n] = 1
        adj[i, (i + 1) % n] = 1
    return adj


def make_line_adjacency(n):
    """Build the undirected line graph on n nodes."""
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def make_complete_adjacency(n):
    """Build the complete undirected graph on n nodes."""
    adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    return adj


def make_small_world_adjacency(n, k=1, p=0.45, seed=0):
    """Build a connected small-world graph by rewiring a ring graph."""
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
    """Return the symmetric Metropolis mixing matrix for an undirected graph."""
    adj = np.asarray(adj, dtype=int)
    degrees = adj.sum(axis=1)
    n = adj.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.flatnonzero(adj[i]):
            W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def make_directed_cycle_weights(n, forward_weights=None):
    """Build a simple row-stochastic directed cycle used to break DGD."""
    if n < 2:
        return np.ones((n, n), dtype=float)

    if forward_weights is None:
        forward_weights = np.linspace(0.2, 0.45, n, dtype=float)
    forward_weights = np.asarray(forward_weights, dtype=float)
    if forward_weights.shape != (n,):
        raise ValueError("forward_weights must have shape (n,).")
    if np.any((forward_weights <= 0.0) | (forward_weights >= 1.0)):
        raise ValueError("Each forward weight must lie strictly between 0 and 1.")

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, i] = 1.0 - forward_weights[i]
        W[i, (i + 1) % n] = forward_weights[i]
    return W


def build_push_sum_column_matrix(graph):
    """Build a column-stochastic push-sum matrix from a directed support graph."""
    graph = np.asarray(graph)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be a square matrix.")

    if np.all(np.isin(graph, [0, 1])):
        support = graph.astype(int)
    else:
        support = (graph > 0).astype(int)
        np.fill_diagonal(support, np.maximum(np.diag(support), 1))

    n = support.shape[0]
    P = np.zeros((n, n), dtype=float)
    for sender in range(n):
        receivers = np.flatnonzero(support[:, sender])
        if sender not in receivers:
            receivers = np.unique(np.append(receivers, sender))
        weight = 1.0 / float(len(receivers))
        P[receivers, sender] = weight
    return P


def spectral_beta(W):
    """Compute the consensus contraction factor |lambda_2(W-J)|."""
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    J = np.ones((n, n), dtype=float) / float(n)
    eigvals = np.linalg.eigvals(W - J)
    return float(np.max(np.abs(eigvals)))


def undirected_edges(adj):
    adj = np.asarray(adj, dtype=int)
    return [(i, j) for i in range(adj.shape[0]) for j in range(i + 1, adj.shape[1]) if adj[i, j] == 1]


def incidence_matrix(adj):
    """Return the node-edge incidence matrix of an undirected graph."""
    edges = undirected_edges(adj)
    B = np.zeros((adj.shape[0], len(edges)), dtype=float)
    for edge_id, (i, j) in enumerate(edges):
        B[i, edge_id] = 1.0
        B[j, edge_id] = -1.0
    return B, edges


def get_graph(name, n, seed=0):
    """Return an adjacency matrix and its default mixing matrix."""
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
    """Compute the dual smoothness constant used for dual decomposition."""
    B = _as_incidence(graph)
    m = agents[0]["H"].shape[0]
    H_inv = _block_diag([agent["H_inv"] for agent in agents])
    B_big = np.kron(B, np.eye(m))
    eigvals = np.linalg.eigvalsh(B_big.T @ H_inv @ B_big)
    return float(max(np.max(eigvals), 1e-12))


def run_dgd(agents, W, alpha_star, step, n_iters, x0=None, seed=0, random_init=False):
    """Run decentralized gradient descent with a fixed mixing matrix."""
    W = np.asarray(W, dtype=float)
    alphas = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
    history = _init_history()
    for _ in range(n_iters):
        grads = grad_all(agents, alphas)
        alphas = W @ alphas - step * grads
        _record_history(history, alphas, alpha_star)
    return _finalize_history(history, alphas)


def run_gradient_tracking(agents, W, alpha_star, step, n_iters, x0=None, seed=0, random_init=False):
    """Run gradient tracking with a fixed mixing matrix."""
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


def random_loss_mixing(adj, p_loss, rng):
    """Sample a Metropolis matrix after randomly dropping undirected links."""
    adj = np.asarray(adj, dtype=int).copy()
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1 and rng.random() < p_loss:
                adj[i, j] = 0
                adj[j, i] = 0
    return metropolis_weights(adj)


def run_dgd_packet_loss(agents, adj, alpha_star, step, n_iters, p_loss=0.3, seed=0, x0=None, random_init=True):
    """Run DGD under random packet losses modeled as edge drops."""
    rng = np.random.default_rng(seed)
    alphas = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
    history = _init_history()
    for _ in range(n_iters):
        W_t = random_loss_mixing(adj, p_loss=p_loss, rng=rng)
        grads = grad_all(agents, alphas)
        alphas = W_t @ alphas - step * grads
        _record_history(history, alphas, alpha_star)
    return _finalize_history(history, alphas)


def run_async_dgd(agents, W, alpha_star, step, n_iters, p_active=0.35, seed=0, x0=None, random_init=True):
    """Run a partially asynchronous DGD variant with random active agents."""
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=float)
    alphas = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
    history = _init_history()
    for _ in range(n_iters):
        grads = grad_all(agents, alphas)
        alphas_mix = W @ alphas
        active = rng.random(len(agents)) < p_active
        alphas_next = alphas_mix.copy()
        alphas_next[active] -= step * grads[active]
        alphas = alphas_next
        _record_history(history, alphas, alpha_star)
    return _finalize_history(history, alphas)


def run_push_sum_dgd(agents, P_col, alpha_star, step, n_iters, x0=None, seed=0, random_init=True):
    """Run push-sum DGD on a directed graph with column-stochastic communication."""
    P_col = np.asarray(P_col, dtype=float)
    X = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
    w = np.ones(len(agents), dtype=float)
    Z = X / w[:, None]

    history = _init_history()
    for _ in range(n_iters):
        grads = grad_all(agents, Z)
        X = P_col @ X - step * grads
        w = P_col @ w
        Z = X / np.maximum(w[:, None], 1e-16)
        _record_history(history, Z, alpha_star)

    history = _finalize_history(history, Z)
    history["push_sum_weights"] = w
    history["raw_states"] = X
    return history


def run_dual_decomposition(agents, graph, alpha_star, step, n_iters):
    """Run the dual decomposition method on an undirected graph."""
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
    """Run edge-based consensus ADMM on an undirected graph."""
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
    """Clip each row to a prescribed Euclidean norm."""
    gradients = np.asarray(gradients, dtype=float)
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / (norms + 1e-12))
    return gradients * scale


# def run_dgd_dp(
#     agents,
#     W,
#     alpha_star,
#     step,
#     epsilon,
#     n_iters,
#     delta=1e-5,
#     clip_norm=5.0,
#     x0=None,
#     seed=0,
#     random_init=False,
# ):
#     """Run the noisy clipped DGD-DP baseline used in Part III."""
#     rng = np.random.default_rng(seed)
#     W = np.asarray(W, dtype=float)
#     alphas = _init_alphas(len(agents), alpha_star.size, x0=x0, seed=seed, random_init=random_init)
#     sensitivity = 2.0 * clip_norm
#     noise_std = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / max(float(epsilon), 1e-12)
#     noise_std /= np.sqrt(max(1, n_iters))

#     history = _init_history()
#     for _ in range(n_iters):
#         grads = clip_rows(grad_all(agents, alphas), clip_norm=clip_norm)
#         noise = rng.normal(0.0, noise_std, size=grads.shape)
#         alphas = W @ alphas - step * (grads + noise)
#         _record_history(history, alphas, alpha_star)

#     history = _finalize_history(history, alphas)
#     history["noise_std"] = float(noise_std)
#     return history

def run_dgd_dp(
    agents,
    W,
    alpha_star,
    step, # Ce paramètre devient le 'step_initial' pour la décroissance
    epsilon,
    n_iters,
    delta,
    clip_norm=1.0, # Sensibilité C pour le clipping L1
    #x0=None,
    seed=0,
    #random_init=False,
):
    """
    le Théorème 2 de l'article [arXiv:2202.01113].
    bruit de Laplace et budget cumulé.
    """
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    
    # initialisation
    alphas = _init_alphas(n_agents, m, x0=None, seed=seed, random_init=False)
    
    # --- Paramètres du Théorème 2 ---
    # C est la borne du gradient. Avec clip_rows (L1), C = 2 * clip_norm
    C_sens = 2.0 * clip_norm
    
    # Échelle du bruit de Laplace (nu) pour un budget epsilon total sur n_iters
    # nu_k = (C * T) / epsilon 
    nu_k = (C_sens * n_iters) / max(float(epsilon), 1e-12)

    # Initialisation de l'historique via la fonction du binôme
    history = _init_history() 

    for k in range(n_iters):
        # 1. Pas décroissants (Condition nécessaire du Théorème 2)
        # On utilise le 'step' passé en argument comme base
        eta_k = 0.002 / (1 + 0.001*k)
        gamma_k = 1 / (1 + 0.001*(k**0.9))
        
        # 2. Calcul des gradients 
        #  On s'assure que clip_rows utilise la norme L1 pour être raccord avec C
        grads = clip_rows(grad_all(agents, alphas), clip_norm=clip_norm)
        
        # 3. Génération du bruit de Laplace 
        noise = rng.laplace(0.0, nu_k, size=grads.shape)
        
        # 4. Mise à jour DGD-DP (Mélange + Gradient bruité)
        # x^{k+1} = x^k + gamma_k * (W - I)(x^k + noise) - eta_k * grad
        noisy_alphas = alphas + noise
        consensus_term = (W @ noisy_alphas) - alphas
        
        alphas = alphas + gamma_k * consensus_term - eta_k * grads
        
        
        _record_history(history, alphas, alpha_star)

    
    history = _finalize_history(history, alphas)
    history["noise_std"] = float(nu_k)
    return history





# ---------------------------------------------------------------------------
# Federated learning
# ---------------------------------------------------------------------------


def build_federated_problem(X, Y, m=10, sigma=SIGMA, nu=NU, landmarks=None):
    """Build the quadratic federated problem used in Part II."""
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
    """Run the FedAvg baseline with optional diminishing stepsizes."""
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
    """Run the optional SCAFFOLD baseline for heterogeneous data."""
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
    """Save a log-log history plot for one or more curves."""
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
    """Save a semi-log history plot for one or more curves."""
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
    """Save one panel per method with the per-agent optimality gaps."""
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
    """Save a scatter plot of the dataset and the Nyström landmarks."""
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


def save_xy_plot(series, path, ylabel, title, xlabel, xscale="linear", yscale="linear"):
    """Save a generic x-y plot for scaling diagnostics."""
    plt.figure(figsize=(6.8, 4.6))
    for label, values in series.items():
        if isinstance(values, dict):
            x = np.asarray(values["x"], dtype=float)
            y = np.asarray(values["y"], dtype=float)
        else:
            x, y = values
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
        if yscale == "log":
            y = np.maximum(y, 1e-16)
        plt.plot(x, y, marker="o", lw=1.8, label=label)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(path))
    plt.savefig(path)
    plt.close()


def save_reconstruction_plot(alpha_methods, x_train, y_train, x_query, x_landmarks, path, title):
    """Save reconstructed functions obtained by several methods."""
    plt.figure(figsize=(8.2, 4.8))
    plt.scatter(x_train, y_train, s=18, alpha=0.45, label="Training points")
    for label, alpha in alpha_methods.items():
        lw = 2.2 if "Centralized" in label else 1.7
        plt.plot(x_query, predict(alpha, x_query, x_landmarks), lw=lw, label=label)
    plt.grid(True, alpha=0.35)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.title(title)
    plt.legend(ncol=2)
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


def first_index_below(curve, threshold):
    """Return the first iterate index below a target threshold."""
    curve = np.asarray(curve, dtype=float)
    idx = np.where(curve <= threshold)[0]
    return int(idx[0]) if len(idx) > 0 else np.nan


def choose_n_agents(n, min_agents=5, max_agents=100, target_local_size=20000):
    """Choose a practical number of agents when scaling n."""
    suggested = int(np.ceil(n / float(target_local_size)))
    suggested = max(min_agents, suggested)
    suggested = min(max_agents, suggested)
    suggested = min(max(1, n), suggested)
    return int(suggested)


def compute_model_for_n(
    x_data,
    y_data,
    n_cur,
    *,
    sigma,
    nu,
    min_agents=5,
    max_agents=100,
    target_local_size=20000,
    seed_landmarks=0,
    batch_size=2048,
):
    """Build the large-n quadratic model while keeping m = ceil(sqrt(n))."""
    n_cur = int(n_cur)
    m_cur = int(np.ceil(np.sqrt(n_cur)))
    n_agents = choose_n_agents(
        n_cur,
        min_agents=min_agents,
        max_agents=max_agents,
        target_local_size=target_local_size,
    )
    x_n = np.asarray(x_data[:n_cur], dtype=float)
    y_n = np.asarray(y_data[:n_cur], dtype=float)

    rng = np.random.default_rng(seed_landmarks + n_cur)
    landmark_idx = np.sort(rng.choice(np.arange(n_cur), size=m_cur, replace=False))
    x_landmarks = x_n[landmark_idx]

    agents = make_streaming_agent_data(
        x_n,
        y_n,
        x_landmarks,
        n_agents=n_agents,
        sigma=sigma,
        nu=nu,
        batch_size=batch_size,
    )
    alpha_star = centralized_solution_from_agents(agents)
    L_max = max(agent["L"] for agent in agents)
    return {
        "n": n_cur,
        "m": m_cur,
        "n_agents": n_agents,
        "x_landmarks": x_landmarks,
        "alpha_star": alpha_star,
        "agents": agents,
        "L_max": float(L_max),
    }


def evaluate_n(
    x_data,
    y_data,
    n_cur,
    *,
    sigma,
    nu,
    threshold,
    T_probe,
    batch_size,
    per_eval_time_limit,
    min_agents,
    max_agents,
    target_local_size,
    seed,
):
    """Probe whether a given n is numerically feasible under a time budget."""
    t0 = time.perf_counter()
    info = {"n": int(n_cur), "m": int(np.ceil(np.sqrt(n_cur)))}
    try:
        model = compute_model_for_n(
            x_data,
            y_data,
            n_cur,
            sigma=sigma,
            nu=nu,
            min_agents=min_agents,
            max_agents=max_agents,
            target_local_size=target_local_size,
            seed_landmarks=10,
            batch_size=batch_size,
        )
        agents = model["agents"]
        alpha_star = model["alpha_star"]

        T_eff = max(20, min(T_probe, int(12000 / max(1, model["m"]))))
        W = metropolis_weights(make_complete_adjacency(model["n_agents"]))
        beta = spectral_beta(W)
        eta_dgd = min(0.9 * (1.0 - beta) / model["L_max"], 0.9 / model["L_max"])
        eta_gt = 0.2 / model["L_max"]

        hist_dgd = run_dgd(agents, W, alpha_star, step=eta_dgd, n_iters=T_eff, seed=seed)
        hist_gt = run_gradient_tracking(agents, W, alpha_star, step=eta_gt, n_iters=T_eff, seed=seed)

        mean_dgd = hist_dgd["mean_gap"]
        mean_gt = hist_gt["mean_gap"]
        finite = bool(np.isfinite(mean_dgd).all() and np.isfinite(mean_gt).all())
        stable = bool(
            (mean_dgd[-1] <= 10.0 * max(mean_dgd[0], 1e-16))
            and (mean_gt[-1] <= 10.0 * max(mean_gt[0], 1e-16))
        )

        elapsed = float(time.perf_counter() - t0)
        feasible = finite and stable and (elapsed <= per_eval_time_limit)
        info.update(
            {
                "m": model["m"],
                "n_agents": model["n_agents"],
                "T_eff": int(T_eff),
                "beta": float(beta),
                "elapsed_s": elapsed,
                "dgd_final": float(mean_dgd[-1]),
                "gt_final": float(mean_gt[-1]),
                "dgd_consensus_final": float(hist_dgd["consensus_gap"][-1]),
                "gt_consensus_final": float(hist_gt["consensus_gap"][-1]),
                "dgd_grad_norm_final": aggregate_gradient_norm(hist_dgd["alpha_mean"], agents),
                "gt_grad_norm_final": aggregate_gradient_norm(hist_gt["alpha_mean"], agents),
                "dgd_it": first_index_below(mean_dgd, threshold),
                "gt_it": first_index_below(mean_gt, threshold),
                "feasible": bool(feasible),
                "reason": "ok" if feasible else ("nan_or_inf" if not finite else ("unstable" if not stable else "too_slow")),
            }
        )
        cached_model = {
            "n": model["n"],
            "m": model["m"],
            "n_agents": model["n_agents"],
            "x_landmarks": model["x_landmarks"],
            "alpha_star": model["alpha_star"],
        }
        return feasible, info, cached_model
    except MemoryError:
        elapsed = float(time.perf_counter() - t0)
        info.update({"elapsed_s": elapsed, "feasible": False, "reason": "memory_error"})
        return False, info, None
    except np.linalg.LinAlgError:
        elapsed = float(time.perf_counter() - t0)
        info.update({"elapsed_s": elapsed, "feasible": False, "reason": "linear_algebra_error"})
        return False, info, None
    except Exception as exc:
        elapsed = float(time.perf_counter() - t0)
        info.update({"elapsed_s": elapsed, "feasible": False, "reason": f"error:{type(exc).__name__}"})
        return False, info, None


def find_largest_n_possible(
    x_data,
    y_data,
    *,
    n_min,
    n_max,
    sigma,
    nu,
    threshold,
    T_probe=120,
    growth=2.0,
    max_evals=14,
    batch_size=2048,
    per_eval_time_limit=90.0,
    total_time_budget=900.0,
    min_agents=5,
    max_agents=100,
    target_local_size=20000,
    seed=0,
):
    """Search the largest feasible n by exponential growth and binary refinement."""
    t_global = time.perf_counter()
    logs = []
    model_cache = {}

    eval_count = 0
    best_n = None
    first_fail_n = None
    n_cur = int(n_min)

    while eval_count < max_evals and n_cur <= n_max:
        if time.perf_counter() - t_global > total_time_budget:
            break

        feasible, info, model = evaluate_n(
            x_data,
            y_data,
            n_cur,
            sigma=sigma,
            nu=nu,
            threshold=threshold,
            T_probe=T_probe,
            batch_size=batch_size,
            per_eval_time_limit=per_eval_time_limit,
            min_agents=min_agents,
            max_agents=max_agents,
            target_local_size=target_local_size,
            seed=seed,
        )
        logs.append(info)
        eval_count += 1
        print(
            f"[eval {eval_count:02d}] n={info['n']}, m={info['m']}, agents={info.get('n_agents', 'na')}, "
            f"feasible={info['feasible']}, reason={info['reason']}, elapsed={info['elapsed_s']:.2f}s"
        )

        if feasible:
            best_n = n_cur
            model_cache[n_cur] = model
            if n_cur == n_max:
                return best_n, logs, model_cache
            n_next = int(max(n_cur + 1, np.floor(n_cur * growth)))
            n_cur = min(n_next, n_max)
        else:
            first_fail_n = n_cur
            break

    if best_n is None:
        raise RuntimeError("No feasible n found from the starting point. Lower n_min or relax limits.")
    if first_fail_n is None:
        return best_n, logs, model_cache

    lo = best_n
    hi = first_fail_n
    while eval_count < max_evals and (hi - lo) > 1:
        if time.perf_counter() - t_global > total_time_budget:
            break

        mid = (lo + hi) // 2
        feasible, info, model = evaluate_n(
            x_data,
            y_data,
            mid,
            sigma=sigma,
            nu=nu,
            threshold=threshold,
            T_probe=T_probe,
            batch_size=batch_size,
            per_eval_time_limit=per_eval_time_limit,
            min_agents=min_agents,
            max_agents=max_agents,
            target_local_size=target_local_size,
            seed=seed,
        )
        logs.append(info)
        eval_count += 1
        print(
            f"[eval {eval_count:02d}] n={info['n']}, m={info['m']}, agents={info.get('n_agents', 'na')}, "
            f"feasible={info['feasible']}, reason={info['reason']}, elapsed={info['elapsed_s']:.2f}s"
        )

        if feasible:
            lo = mid
            best_n = mid
            model_cache[mid] = model
        else:
            hi = mid

    return best_n, logs, model_cache


def tune_best_histories(agents, alpha_star, W, *, eta_multipliers, iteration_budget, seed=0):
    """Retune DGD and GT at the largest feasible n and keep the best curves."""
    m = alpha_star.size
    L_max = max(agent["L"] for agent in agents)
    beta = spectral_beta(W)
    base_eta = min((1.0 - beta) / L_max, 1.0 / L_max)
    T_long = max(300, int(iteration_budget / max(1, m)))

    best_dgd = {"final": np.inf}
    best_gt = {"final": np.inf}
    for mult in eta_multipliers:
        eta_try = float(mult) * base_eta
        hist_dgd = run_dgd(agents, W, alpha_star, step=eta_try, n_iters=T_long, seed=seed)
        hist_gt = run_gradient_tracking(agents, W, alpha_star, step=eta_try, n_iters=T_long, seed=seed)

        final_dgd = float(hist_dgd["mean_gap"][-1])
        final_gt = float(hist_gt["mean_gap"][-1])
        if np.isfinite(final_dgd) and final_dgd < best_dgd["final"]:
            best_dgd = {"eta": eta_try, "history": hist_dgd, "final": final_dgd}
        if np.isfinite(final_gt) and final_gt < best_gt["final"]:
            best_gt = {"eta": eta_try, "history": hist_gt, "final": final_gt}

    return best_dgd, best_gt, T_long


def select_scaling_plot_ns(records, max_points=6):
    """Subsample feasible n values for readable reconstruction plots."""
    feasible_ns = [record["n"] for record in records]
    if len(feasible_ns) <= max_points:
        return feasible_ns
    idx = np.linspace(0, len(feasible_ns) - 1, max_points, dtype=int)
    return sorted(set(feasible_ns[i] for i in idx))


def part1_steps(agents, W, incidence):
    """Compute theory-motivated stepsizes and penalty parameters for Part I."""
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


def write_part1_summary(path, records, line_steps, scaling_records, n_best):
    """Write a compact theory-and-checklist summary alongside the figures."""
    lines = [
        "Part I checklist",
        "================",
        "",
        "1. Baseline distributed methods use the course assumptions: connected undirected graph and doubly-stochastic mixing.",
        f"   DGD step uses min(0.9*(1-beta)/L, 0.9/L) with beta={line_steps['beta']:.4f} and L_max={line_steps['L_max']:.4f}.",
        f"   GT step uses 0.2/L_max = {line_steps['eta_gt']:.3e}.",
        f"   Dual decomposition uses tau = 1/(2*L_dual) = {line_steps['tau_dual']:.3e}.",
        f"   ADMM uses rho = sqrt(mu_min*L_max) = {line_steps['rho_admm']:.3e}.",
        "",
        "2. Directed communication breaks the DGD proof because the average iterate is no longer preserved when the matrix is not doubly stochastic.",
        "   Packet losses create a time-varying graph that may disconnect the network and violate the fixed-mixing assumption.",
        "   Asynchrony violates the synchronous update model used in the standard contraction argument.",
        "",
        "3. Push-sum restores the missing Perron normalization on directed graphs by tracking weights w_i^t and using z_i^t = x_i^t / w_i^t.",
        "",
        "4. Large-n scaling keeps m = ceil(sqrt(n)).",
        "   The scaling suite also logs consensus gaps and aggregate gradient norms, so it still gives convergence indicators even if a reference solution becomes unavailable.",
        f"   Largest feasible n found under the current compute budget: {n_best}.",
        "",
        "Generated figures",
        "-----------------",
    ]
    for name in records:
        lines.append(f"- {name}")
    if scaling_records:
        lines.append("")
        lines.append("Feasible scaling records")
        lines.append("------------------------")
        for rec in scaling_records:
            lines.append(
                f"n={rec['n']}, m={rec['m']}, agents={rec['n_agents']}, "
                f"dgd_final={rec['dgd_final']:.3e}, gt_final={rec['gt_final']:.3e}"
            )

    ensure_dir(str(path.parent))
    with open(path, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")


def run_line_baselines(problem, agents, alpha_star):
    """Run the four baseline methods on the line graph and save core figures."""
    adj_line, W_line = get_graph("line", N_AGENTS, seed=SEED)
    incidence_line, _ = incidence_matrix(adj_line)
    line_steps = part1_steps(agents, W_line, incidence_line)

    histories = {
        "DGD": run_dgd(agents, W_line, alpha_star, step=line_steps["eta_dgd"], n_iters=PART1_DGD_ITERS, seed=SEED),
        "Gradient tracking": run_gradient_tracking(
            agents,
            W_line,
            alpha_star,
            step=line_steps["eta_gt"],
            n_iters=PART1_GT_ITERS,
            seed=SEED,
        ),
        "Dual decomposition": run_dual_decomposition(
            agents,
            incidence_line,
            alpha_star,
            step=line_steps["tau_dual"],
            n_iters=PART1_DD_ITERS,
        ),
        "ADMM": run_consensus_admm(
            agents,
            adj_line,
            alpha_star,
            rho=line_steps["rho_admm"],
            n_iters=PART1_ADMM_ITERS,
        ),
    }

    save_agent_gap_grid_plot(
        {label: hist["agent_gaps"] for label, hist in histories.items()},
        FIGURES_DIR / "part1_gap_line.pdf",
        ylabel=r"$\|\alpha_i^t-\alpha^\star\|$",
        title="Part I - line graph",
    )

    x_query = np.linspace(-1.0, 1.0, 250)
    alpha_methods = {"Centralized": alpha_star}
    for label, hist in histories.items():
        alpha_methods[label] = hist["alpha_mean"]
    save_reconstruction_plot(
        alpha_methods,
        problem["x_n"],
        problem["y_n"],
        x_query,
        problem["x_m"],
        FIGURES_DIR / "part1_reconstruction_compare.pdf",
        title="Centralized vs distributed reconstruction",
    )
    return histories, adj_line, W_line, incidence_line, line_steps


def run_graph_sweep(agents, alpha_star):
    """Compare the baseline methods across the graph families from the subject."""
    histories_by_method = {
        "DGD": {},
        "Gradient tracking": {},
        "Dual decomposition": {},
        "ADMM": {},
    }
    method_to_file = {
        "DGD": "part1_dgd_graph_compare.pdf",
        "Gradient tracking": "part1_gt_graph_compare.pdf",
        "Dual decomposition": "part1_dual_graph_compare.pdf",
        "ADMM": "part1_admm_graph_compare.pdf",
    }

    for graph_name, graph_label in GRAPH_SPECS:
        adj, W = get_graph(graph_name, N_AGENTS, seed=SEED)
        incidence, _ = incidence_matrix(adj)
        params = part1_steps(agents, W, incidence)

        histories_by_method["DGD"][graph_label] = run_dgd(
            agents,
            W,
            alpha_star,
            step=params["eta_dgd"],
            n_iters=PART1_DGD_ITERS,
            seed=SEED,
        )["bar_gap"]
        histories_by_method["Gradient tracking"][graph_label] = run_gradient_tracking(
            agents,
            W,
            alpha_star,
            step=params["eta_gt"],
            n_iters=PART1_GT_ITERS,
            seed=SEED,
        )["bar_gap"]
        histories_by_method["Dual decomposition"][graph_label] = run_dual_decomposition(
            agents,
            incidence,
            alpha_star,
            step=params["tau_dual"],
            n_iters=PART1_DD_ITERS,
        )["bar_gap"]
        histories_by_method["ADMM"][graph_label] = run_consensus_admm(
            agents,
            adj,
            alpha_star,
            rho=params["rho_admm"],
            n_iters=PART1_ADMM_ITERS,
        )["bar_gap"]
        print(
            f"  [{graph_label}] beta={params['beta']:.6f}, "
            f"eta_dgd={params['eta_dgd']:.3e}, eta_gt={params['eta_gt']:.3e}, "
            f"tau_dual={params['tau_dual']:.3e}, rho_admm={params['rho_admm']:.3e}"
        )

    for method, curves in histories_by_method.items():
        save_history_plot(
            curves,
            FIGURES_DIR / method_to_file[method],
            ylabel=r"$\|\bar{\alpha}^t-\alpha^\star\|$",
            title=f"{method} - graph effect on the averaged iterate",
        )
    return histories_by_method


def run_break_and_push_sum(agents, alpha_star, adj_line, W_line, line_steps):
    """Run the convergence-breaking experiments and the push-sum recovery test."""
    W_directed = make_directed_cycle_weights(len(agents))
    P_col = build_push_sum_column_matrix(W_directed)

    baseline = run_dgd(
        agents,
        W_line,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        seed=SEED + 1,
        random_init=True,
    )
    directed = run_dgd(
        agents,
        W_directed,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        seed=SEED + 1,
        random_init=True,
    )
    packet_loss = run_dgd_packet_loss(
        agents,
        adj_line,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        p_loss=PART1_PACKET_LOSS,
        seed=SEED + 1,
    )
    asynchronous = run_async_dgd(
        agents,
        W_line,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        p_active=PART1_ACTIVE_PROB,
        seed=SEED + 1,
    )

    save_history_plot(
        {
            "Undirected baseline": baseline["mean_gap"],
            "Directed": directed["mean_gap"],
            "Packet losses": packet_loss["mean_gap"],
            "Asynchronous": asynchronous["mean_gap"],
        },
        FIGURES_DIR / "part1_break_convergence.pdf",
        ylabel="Mean optimality gap",
        title="Breaking convergence scenarios",
    )

    eta_push_sum = min(0.45 / line_steps["L_max"], 0.95 / line_steps["L_max"])
    push_sum = run_push_sum_dgd(
        agents,
        P_col,
        alpha_star,
        step=eta_push_sum,
        n_iters=PART1_BREAK_ITERS,
        seed=SEED + 1,
    )
    save_history_plot(
        {
            "Directed DGD": directed["mean_gap"],
            "Push-sum DGD": push_sum["mean_gap"],
        },
        FIGURES_DIR / "part1_push_sum_recovery.pdf",
        ylabel="Mean optimality gap",
        title="Directed communication: push-sum recovery",
    )

    print(f"  directed columns sums = {np.round(W_directed.sum(axis=0), 4)}")
    print(f"  push-sum column sums = {np.round(P_col.sum(axis=0), 4)}")
    print(
        f"  break suite final gaps: baseline={baseline['mean_gap'][-1]:.3e}, "
        f"directed={directed['mean_gap'][-1]:.3e}, "
        f"loss={packet_loss['mean_gap'][-1]:.3e}, async={asynchronous['mean_gap'][-1]:.3e}, "
        f"push-sum={push_sum['mean_gap'][-1]:.3e}"
    )


def run_scaling_suite(x, y):
    """Run the large-n search and save surrogate convergence diagnostics."""
    threshold = PART1_SCALING_THRESHOLD
    n_best, search_logs, model_cache = find_largest_n_possible(
        x,
        y,
        n_min=100,
        n_max=len(x),
        sigma=SIGMA,
        nu=NU,
        threshold=threshold,
        T_probe=PART1_SCALING_PROBE_ITERS,
        growth=2.0,
        max_evals=PART1_SCALING_MAX_EVALS,
        batch_size=PART1_SCALING_BATCH_SIZE,
        per_eval_time_limit=PART1_SCALING_PER_EVAL_LIMIT_S,
        total_time_budget=PART1_SCALING_TOTAL_BUDGET_S,
        min_agents=PART1_SCALING_MIN_AGENTS,
        max_agents=PART1_SCALING_MAX_AGENTS,
        target_local_size=PART1_SCALING_TARGET_LOCAL_SIZE,
        seed=SEED,
    )
    records = sorted([record for record in search_logs if record["feasible"]], key=lambda record: record["n"])

    save_xy_plot(
        {
            "DGD final gap": ([record["n"] for record in records], [record["dgd_final"] for record in records]),
            "GT final gap": ([record["n"] for record in records], [record["gt_final"] for record in records]),
        },
        FIGURES_DIR / "part1_scaling_final_gap.pdf",
        ylabel="Final mean optimality gap",
        title="Convergence quality vs n",
        xlabel="n",
        xscale="log",
        yscale="log",
    )
    save_xy_plot(
        {
            "DGD iters to threshold": ([record["n"] for record in records], [record["dgd_it"] for record in records]),
            "GT iters to threshold": ([record["n"] for record in records], [record["gt_it"] for record in records]),
        },
        FIGURES_DIR / "part1_scaling_iterations.pdf",
        ylabel=f"Iterations to mean gap <= {threshold}",
        title="Speed vs n",
        xlabel="n",
        xscale="log",
        yscale="linear",
    )
    save_xy_plot(
        {
            "DGD final consensus gap": (
                [record["n"] for record in records],
                [record["dgd_consensus_final"] for record in records],
            ),
            "GT final consensus gap": (
                [record["n"] for record in records],
                [record["gt_consensus_final"] for record in records],
            ),
            "DGD final aggregate grad norm": (
                [record["n"] for record in records],
                [record["dgd_grad_norm_final"] for record in records],
            ),
            "GT final aggregate grad norm": (
                [record["n"] for record in records],
                [record["gt_grad_norm_final"] for record in records],
            ),
        },
        FIGURES_DIR / "part1_scaling_surrogates.pdf",
        ylabel="Surrogate convergence metrics",
        title="Consensus and stationarity vs n",
        xlabel="n",
        xscale="log",
        yscale="log",
    )

    selected_ns = select_scaling_plot_ns(records)
    x_query = np.linspace(-1.0, 1.0, 250)
    curves = {}
    for n_cur in selected_ns:
        model = model_cache.get(n_cur)
        if model is None:
            model = compute_model_for_n(
                x,
                y,
                n_cur,
                sigma=SIGMA,
                nu=NU,
                min_agents=PART1_SCALING_MIN_AGENTS,
                max_agents=PART1_SCALING_MAX_AGENTS,
                target_local_size=PART1_SCALING_TARGET_LOCAL_SIZE,
                seed_landmarks=10,
                batch_size=PART1_SCALING_BATCH_SIZE,
            )
        curves[f"n={n_cur}"] = predict(model["alpha_star"], x_query, model["x_landmarks"])

    plt.figure(figsize=(8.4, 4.8))
    for label, values in curves.items():
        plt.plot(x_query, values, lw=1.8, label=label)
    plt.grid(True, alpha=0.35)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.title("Centralized reconstructed functions for selected n")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "part1_scaling_functions.pdf")
    plt.close()

    best_model = compute_model_for_n(
        x,
        y,
        n_best,
        sigma=SIGMA,
        nu=NU,
        min_agents=PART1_SCALING_MIN_AGENTS,
        max_agents=PART1_SCALING_MAX_AGENTS,
        target_local_size=PART1_SCALING_TARGET_LOCAL_SIZE,
        seed_landmarks=11,
        batch_size=max(PART1_SCALING_BATCH_SIZE, 4096),
    )
    W_best = metropolis_weights(make_complete_adjacency(best_model["n_agents"]))
    best_dgd, best_gt, T_long = tune_best_histories(
        best_model["agents"],
        best_model["alpha_star"],
        W_best,
        eta_multipliers=PART1_SCALING_ETA_MULTIPLIERS,
        iteration_budget=PART1_SCALING_LONG_ITER_BUDGET,
        seed=SEED + 2,
    )

    save_history_plot(
        {
            f"DGD tuned (eta={best_dgd['eta']:.2e})": best_dgd["history"]["mean_gap"],
            f"GT tuned (eta={best_gt['eta']:.2e})": best_gt["history"]["mean_gap"],
        },
        FIGURES_DIR / "part1_scaling_best_tuned.pdf",
        ylabel="Mean optimality gap",
        title=f"Improved convergence at the largest feasible n={n_best}",
    )

    print(
        f"  scaling best n={n_best}, m={best_model['m']}, agents={best_model['n_agents']}, "
        f"T_long={T_long}, best_dgd={best_dgd['final']:.3e}, best_gt={best_gt['final']:.3e}"
    )
    return n_best, records


def run_part1():
    """Execute the full Part I experimental suite."""
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

    line_histories, adj_line, W_line, _incidence_line, line_steps = run_line_baselines(problem, agents, alpha_star)
    _graph_histories = run_graph_sweep(agents, alpha_star)
    run_break_and_push_sum(agents, alpha_star, adj_line, W_line, line_steps)
    n_best, scaling_records = run_scaling_suite(x, y)

    print(f"  line graph tau_dual = {line_steps['tau_dual']:.3e}")
    print(f"  line graph rho_admm = {line_steps['rho_admm']:.3e}")
    for label, hist in line_histories.items():
        print(f"  final {label} bar gap = {hist['bar_gap'][-1]:.6e}")

    generated = [
        "part1_dataset.pdf",
        "part1_gap_line.pdf",
        "part1_reconstruction_compare.pdf",
        "part1_dgd_graph_compare.pdf",
        "part1_gt_graph_compare.pdf",
        "part1_dual_graph_compare.pdf",
        "part1_admm_graph_compare.pdf",
        "part1_break_convergence.pdf",
        "part1_push_sum_recovery.pdf",
        "part1_scaling_final_gap.pdf",
        "part1_scaling_iterations.pdf",
        "part1_scaling_surrogates.pdf",
        "part1_scaling_functions.pdf",
        "part1_scaling_best_tuned.pdf",
    ]
    write_part1_summary(
        FIGURES_DIR / "part1_theory_summary.txt",
        generated,
        line_steps,
        scaling_records,
        n_best,
    )


# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------


def run_part2():
    """Execute the full Part II federated-learning suite."""
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
    """Execute the full Part III private DGD suite."""
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
    """Run the three project parts sequentially."""
    print("\n===== Part I =====")
    run_part1()
    print("\n===== Part II =====")
    run_part2()
    print("\n===== Part III =====")
    run_part3()


def main(argv=None):
    """Parse the CLI and dispatch to the requested part."""
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
