import numpy as np

from .data_utils import as_1d
from .distributed_objectives import grad_all, optimality_gap
from .graphs import adjacency_from_weights, incidence_matrix, undirected_edges


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
