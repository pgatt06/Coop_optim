import numpy as np


def cycle_graph_weights(a):
    W = np.zeros((a, a), dtype=float)
    for i in range(a):
        W[i, i] = 0.5
        W[i, (i - 1) % a] = 0.25
        W[i, (i + 1) % a] = 0.25
    return W


def line_graph_weights(a):
    W = np.zeros((a, a), dtype=float)
    for i in range(a):
        W[i, i] = 0.6
        if i > 0:
            W[i, i - 1] = 0.2
        if i < a - 1:
            W[i, i + 1] = 0.2
    W[0, 0] = 0.8
    W[-1, -1] = 0.8
    return W


def complete_graph_weights(a):
    return np.ones((a, a), dtype=float) / float(a)


def adjacency_from_weights(W):
    adj = (np.asarray(W, dtype=float) > 0).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def get_graph(name, n):
    name = name.lower()
    if name == 'cycle':
        W = cycle_graph_weights(n)
    elif name == 'line':
        W = line_graph_weights(n)
    elif name in ('complete', 'fully_connected', 'full'):
        W = complete_graph_weights(n)
    else:
        raise ValueError(f'Unknown graph: {name}')
    return adjacency_from_weights(W), W
