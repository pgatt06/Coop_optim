import numpy as np


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
