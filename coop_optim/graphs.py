import numpy as np


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


def is_strongly_connected(adj):
    """Return True when every node can reach every other node in both directions."""
    adj = np.asarray(adj, dtype=int)
    if adj.shape[0] == 0:
        return True

    def _reachable(matrix):
        seen = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(matrix[node]):
                if neighbor not in seen:
                    seen.add(int(neighbor))
                    stack.append(int(neighbor))
        return len(seen) == matrix.shape[0]

    return _reachable(adj) and _reachable(adj.T)


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
    """List undirected edges once with i < j."""
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
