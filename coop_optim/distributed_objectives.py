import numpy as np

from .centralized import quadratic_form_for_agent
from .data_utils import split_indices_equally


def make_agent_data(problem, n_agents):
    M = problem['M']
    y = problem['y_n']
    Kmm = problem['Kmm']
    splits = split_indices_equally(len(y), n_agents)
    agents = []
    for idx in splits:
        M_i = M[idx]
        y_i = y[idx]
        Q_i, b_i = quadratic_form_for_agent(M_i, y_i, Kmm, n_agents)
        agents.append({
            'indices': idx,
            'M': M_i,
            'y': y_i,
            'Q': Q_i,
            'b': b_i,
        })
    return agents


def local_objective(alpha, agent):
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    return 0.5 * alpha @ (agent['Q'] @ alpha) - agent['b'] @ alpha


def local_gradient(alpha, agent):
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    return agent['Q'] @ alpha - agent['b']


def stacked_average(alphas):
    return np.mean(np.asarray(alphas, dtype=float), axis=0)
