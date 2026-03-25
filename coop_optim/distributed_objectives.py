import numpy as np

from .centralized import quadratic_form_for_agent
from .data_utils import as_1d, split_indices_equally


def make_agent_data(problem, n_agents, sigma=0.5, nu=1.0):
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
