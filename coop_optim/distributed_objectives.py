import numpy as np

from .centralized import quadratic_form_for_agent
from .kernel_utils import cov_matrix, cross_cov_matrix
from .data_utils import as_1d, split_indices_equally


def make_agent_data(problem, n_agents, sigma=0.5, nu=1.0):
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
    """Evaluate the local quadratic gradient at one agent."""
    alpha = as_1d(alpha)
    return agent["H"] @ alpha - agent["b"]


def grad_all(agents, alphas):
    """Stack all local gradients row-wise."""
    alphas = np.asarray(alphas, dtype=float)
    return np.vstack([local_gradient(alphas[i], agents[i]) for i in range(len(agents))])


def optimality_gap(alphas, alpha_star):
    """Compute ||alpha_i - alpha_star|| for every agent."""
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
    """Recover the centralized optimizer directly from local quadratic models."""
    H_total, b_total = aggregate_quadratic_model(agents)
    return np.linalg.solve(H_total, b_total)


def aggregate_gradient(alpha, agents):
    """Evaluate the centralized gradient using only local quadratic summaries."""
    H_total, b_total = aggregate_quadratic_model(agents)
    return H_total @ as_1d(alpha) - b_total


def aggregate_gradient_norm(alpha, agents):
    """Return the norm of the centralized gradient surrogate."""
    return float(np.linalg.norm(aggregate_gradient(alpha, agents)))


def make_streaming_agent_data(x_data, y_data, x_landmarks, n_agents, sigma=0.5, nu=1.0, batch_size=2048):
    """Build local quadratic models in batches without materializing the full Knm matrix."""
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
