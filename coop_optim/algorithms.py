import numpy as np

from .distributed_objectives import local_gradient


def _init_alphas(n_agents, m, x0=None, seed=0, random_init=False):
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        return np.tile(x0, (n_agents, 1))
    if random_init:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1e-3, size=(n_agents, m))
    return np.zeros((n_agents, m), dtype=float)


def _record_opt_gap(history, alphas, alpha_star):
    alpha_star = np.asarray(alpha_star, dtype=float).reshape(-1)
    gaps = np.linalg.norm(alphas - alpha_star.reshape(1, -1), axis=1)
    history['mean_gap'].append(float(np.mean(gaps)))
    history['max_gap'].append(float(np.max(gaps)))
    mean_alpha = np.mean(alphas, axis=0, keepdims=True)
    history['consensus_gap'].append(float(np.max(np.linalg.norm(alphas - mean_alpha, axis=1))))


def run_dgd(agents, W, alpha_star, step, n_iters, x0=None, seed=0, random_init=False):
    """
    DGD :
        alpha_i^{t+1} = sum_j W_ij alpha_j^t - eta * grad f_i(alpha_i^t)
    """
    W = np.asarray(W, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, seed=seed, random_init=random_init)

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for _ in range(n_iters):
        new_alphas = np.zeros_like(alphas)
        for i, agent in enumerate(agents):
            grad_i = local_gradient(alphas[i], agent)
            new_alphas[i] = W[i, :] @ alphas - step * grad_i
        alphas = new_alphas
        _record_opt_gap(history, alphas, alpha_star)

    history['alphas'] = alphas
    history['alpha_mean'] = np.mean(alphas, axis=0)
    return history


def run_gradient_tracking(agents, W, alpha_star, step, n_iters, x0=None, seed=0, random_init=False):
    """
    Suivi de gradient.
    """
    W = np.asarray(W, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, seed=seed, random_init=random_init)

    trackers = np.zeros((n_agents, m), dtype=float)
    for i, agent in enumerate(agents):
        trackers[i] = local_gradient(alphas[i], agent)

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for _ in range(n_iters):
        new_alphas = np.zeros_like(alphas)
        new_trackers = np.zeros_like(trackers)
        for i, agent in enumerate(agents):
            grad_k = local_gradient(alphas[i], agent)
            new_alphas[i] = W[i, :] @ alphas - step * trackers[i]
            grad_k1 = local_gradient(new_alphas[i], agent)
            new_trackers[i] = W[i, :] @ trackers - grad_k + grad_k1
        alphas = new_alphas
        trackers = new_trackers
        _record_opt_gap(history, alphas, alpha_star)

    history['alphas'] = alphas
    history['alpha_mean'] = np.mean(alphas, axis=0)
    return history


def _edge_data_from_W(W):
    W = np.asarray(W, dtype=float)
    undirected_edges = []
    directed_edges = []
    n_agents = W.shape[0]
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and W[i, j] > 0:
                directed_edges.append((i, j))
                if i > j and W[j, i] > 0:
                    undirected_edges.append((i, j))
    return undirected_edges, directed_edges


def run_dual_decomposition(agents, W, alpha_star, step, n_iters, x0=None):
    """
    Décomposition duale pair-à-pair.
    Les variables duales (lambdas) vivent sur les arêtes non orientées (i,j) avec i>j.
    """
    W = np.asarray(W, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, random_init=False)
    undirected_edges, _ = _edge_data_from_W(W)
    lambdas = {edge: np.zeros(m, dtype=float) for edge in undirected_edges}

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for _ in range(n_iters):
        for i, agent in enumerate(agents):
            dual_sum = np.zeros(m, dtype=float)
            for j in range(n_agents):
                if i == j or W[i, j] <= 0:
                    continue
                if j < i and (i, j) in lambdas:
                    dual_sum += lambdas[(i, j)]
                elif j > i and (j, i) in lambdas:
                    dual_sum -= lambdas[(j, i)]
            alphas[i] = np.linalg.solve(agent['Q'], agent['b'] - dual_sum)

        for (i, j) in lambdas:
            lambdas[(i, j)] += step * (alphas[i] - alphas[j])

        _record_opt_gap(history, alphas, alpha_star)

    history['alphas'] = alphas
    history['alpha_mean'] = np.mean(alphas, axis=0)
    history['lambdas'] = lambdas
    return history


def run_consensus_admm(agents, W, alpha_star, rho, n_iters, x0=None):
    """
    ADMM de consensus.
    - lambda_(i,j) et y_(i,j) pour chaque arête orientée
    - alpha_i local résolu exactement à chaque itération
    """
    W = np.asarray(W, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, random_init=False)
    _, directed_edges = _edge_data_from_W(W)

    lambdas = {edge: np.zeros(m, dtype=float) for edge in directed_edges}
    y_edges = {edge: np.zeros(m, dtype=float) for edge in directed_edges}

    inv_A = []
    for i, agent in enumerate(agents):
        degree_i = int(np.sum(W[i, :] > 0) - 1)
        inv_A.append(np.linalg.inv(agent['Q'] + rho * degree_i * np.eye(m)))

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for _ in range(n_iters):
        new_alphas = np.zeros_like(alphas)
        for i, agent in enumerate(agents):
            prox_sum = np.zeros(m, dtype=float)
            for j in range(n_agents):
                if i != j and W[i, j] > 0:
                    prox_sum += y_edges[(i, j)] - lambdas[(i, j)] / rho
            new_alphas[i] = inv_A[i] @ (agent['b'] + rho * prox_sum)

        alphas = new_alphas
        for (i, j) in directed_edges:
            y_edges[(i, j)] = 0.5 * (alphas[i] + alphas[j])
            lambdas[(i, j)] += rho * (alphas[i] - y_edges[(i, j)])

        _record_opt_gap(history, alphas, alpha_star)

    history['alphas'] = alphas
    history['alpha_mean'] = np.mean(alphas, axis=0)
    history['lambdas'] = lambdas
    history['y_edges'] = y_edges
    return history


def get_corrupted_context(W_base, mode='normal', p_loss=0.2, p_active=0.4, seed=None):
    rng = np.random.default_rng(seed)
    W_t = np.asarray(W_base, dtype=float).copy()
    n_agents = W_t.shape[0]
    active_mask = np.ones(n_agents, dtype=bool)

    if mode == 'directed':
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if W_t[i, j] > 0 or W_t[j, i] > 0:
                    if rng.random() < 0.5:
                        W_t[i, j] = 0.0
                    else:
                        W_t[j, i] = 0.0

    elif mode == 'loss':
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and W_t[i, j] > 0 and rng.random() < p_loss:
                    W_t[i, j] = 0.0
        for i in range(n_agents):
            row_sum = np.sum(W_t[i, :])
            if row_sum > 0:
                W_t[i, :] /= row_sum

    elif mode == 'async':
        active_mask = rng.random(n_agents) < p_active
        if not np.any(active_mask):
            active_mask[rng.integers(0, n_agents)] = True

    return W_t, active_mask


def make_directed_row_stochastic(W_base, seed=0):
    """
    Construit une matrice de mélange orientée (ligne-stochastique) à partir d'une matrice
    symétrique/undirected, en supprimant aléatoirement une direction sur chaque lien.
    """
    rng = np.random.default_rng(seed)
    W = np.asarray(W_base, dtype=float).copy()
    n_agents = W.shape[0]

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if W[i, j] > 0 or W[j, i] > 0:
                if rng.random() < 0.5:
                    W[i, j] = 0.0
                else:
                    W[j, i] = 0.0

    for i in range(n_agents):
        row_sum = np.sum(W[i, :])
        if row_sum <= 0:
            W[i, :] = 0.0
            W[i, i] = 1.0
        else:
            W[i, :] /= row_sum
    return W


def _lossy_row_stochastic_weights(W_base, p_loss, rng):
    """
    Applique des pertes de paquets sur les arcs de communication puis renormalise
    chaque ligne pour conserver une matrice ligne-stochastique.
    """
    W = np.asarray(W_base, dtype=float).copy()
    n_agents = W.shape[0]

    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and W[i, j] > 0 and rng.random() < p_loss:
                W[i, j] = 0.0

    for i in range(n_agents):
        row_sum = np.sum(W[i, :])
        if row_sum <= 0:
            W[i, :] = 0.0
            W[i, i] = 1.0
        else:
            W[i, :] /= row_sum
    return W


def run_dgd_packet_loss(
    agents,
    W_base,
    alpha_star,
    step,
    n_iters,
    p_loss=0.3,
    x0=None,
    seed=0,
    random_init=False,
):
    """
    DGD avec pertes de paquets : à chaque itération, on perturbe la matrice de mélange.
    """
    rng = np.random.default_rng(seed)
    W_base = np.asarray(W_base, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, seed=seed, random_init=random_init)

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for _ in range(n_iters):
        W_t = _lossy_row_stochastic_weights(W_base, p_loss=p_loss, rng=rng)
        new_alphas = np.zeros_like(alphas)
        for i, agent in enumerate(agents):
            grad_i = local_gradient(alphas[i], agent)
            new_alphas[i] = W_t[i, :] @ alphas - step * grad_i
        alphas = new_alphas
        _record_opt_gap(history, alphas, alpha_star)

    history['alphas'] = alphas
    history['alpha_mean'] = np.mean(alphas, axis=0)
    return history


def run_dual_decomposition_packet_loss(agents, W_base, alpha_star, step, n_iters, p_loss=0.3, x0=None, seed=0):
    """
    Décomposition Duale avec pertes de paquets.
    Si un paquet est perdu sur l'arête (i,j), la variable duale lambda_ij n'est pas mise à jour.
    """
    rng = np.random.default_rng(seed)
    W_base = np.asarray(W_base, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, random_init=False)
    
    # On identifie les arêtes existantes dans le graphe de base
    undirected_edges, _ = _edge_data_from_W(W_base)
    lambdas = {edge: np.zeros(m, dtype=float) for edge in undirected_edges}

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    
    for _ in range(n_iters):
        # 1. Étape Primale : Toujours la même (résolution exacte locale)
        for i, agent in enumerate(agents):
            dual_sum = np.zeros(m, dtype=float)
            for j in range(n_agents):
                if i == j or W_base[i, j] <= 0:
                    continue
                
                # Simulation de la perte de paquet : si l'arête échoue, 
                # l'agent i ne "voit" pas la contrainte avec j pour cette itération
                if rng.random() > p_loss: 
                    if j < i and (i, j) in lambdas:
                        dual_sum += lambdas[(i, j)]
                    elif j > i and (j, i) in lambdas:
                        dual_sum -= lambdas[(j, i)]
            
            alphas[i] = np.linalg.solve(agent['Q'], agent['b'] - dual_sum)

        # 2. Étape Duale : Mise à jour des multiplicateurs seulement si le lien est actif
        for (i, j) in lambdas:
            if rng.random() > p_loss: # Si le lien fonctionne
                lambdas[(i, j)] += step * (alphas[i] - alphas[j])

        _record_opt_gap(history, alphas, alpha_star)

    return history









def run_async_dgd(
    agents,
    W,
    alpha_star,
    step,
    n_iters,
    p_active=0.35,
    x0=None,
    seed=0,
    random_init=False,
):
    """
    DGD asynchrone : à chaque itération, seul un sous-ensemble d'agents applique
    la mise à jour gradient.
    """
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, seed=seed, random_init=random_init)

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for _ in range(n_iters):
        G = np.vstack([local_gradient(alphas[i], agents[i]) for i in range(n_agents)])
        mixed = W @ alphas
        active = rng.random(n_agents) < p_active
        if not np.any(active):
            active[rng.integers(0, n_agents)] = True
        new_alphas = mixed.copy()
        new_alphas[active] -= step * G[active]
        alphas = new_alphas
        _record_opt_gap(history, alphas, alpha_star)

    history['alphas'] = alphas
    history['alpha_mean'] = np.mean(alphas, axis=0)
    return history


def build_column_stochastic_pushsum(W_row):
    """
    Construit une matrice colonne-stochastique P à partir d'une matrice ligne-stochastique W
    (interprétation push-sum).
    """
    W_row = np.asarray(W_row, dtype=float)
    n_agents = W_row.shape[0]
    out_neighbors = [[] for _ in range(n_agents)]
    for j in range(n_agents):
        for i in range(n_agents):
            if i != j and W_row[i, j] > 0:
                out_neighbors[j].append(i)

    P = np.zeros((n_agents, n_agents), dtype=float)
    for j in range(n_agents):
        receivers = [j] + out_neighbors[j]
        weight = 1.0 / len(receivers)
        for i in receivers:
            P[i, j] = weight
    return P


def run_push_sum_dgd_directed(
    agents,
    P_col,
    alpha_star,
    step,
    n_iters,
    x0=None,
    seed=0,
    random_init=False,
):
    """
    DGD Push-Sum pour graphe orienté (matrice colonne-stochastique P_col).
    """
    rng = np.random.default_rng(seed)
    P_col = np.asarray(P_col, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size

    if x0 is not None:
        base = np.asarray(x0, dtype=float).reshape(-1)
        X = np.tile(base, (n_agents, 1))
    elif random_init:
        X = rng.normal(0.0, 1e-3, size=(n_agents, m))
    else:
        X = np.zeros((n_agents, m), dtype=float)

    w = np.ones(n_agents, dtype=float)
    Z = X / w[:, None]

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for _ in range(n_iters):
        G = np.vstack([local_gradient(Z[i], agents[i]) for i in range(n_agents)])
        X = P_col @ X - step * G
        w = np.clip(P_col @ w, 1e-12, None)
        Z = X / w[:, None]
        _record_opt_gap(history, Z, alpha_star)

    history['alphas'] = Z
    history['alpha_mean'] = np.mean(Z, axis=0)
    return history


def run_dgd_dp(
    agents,
    W,
    alpha_star,
    epsilon,
    n_iters,
    sensitivity=0.5,
    x0=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size
    alphas = _init_alphas(n_agents, m, x0=x0, random_init=False)
    b_scale = sensitivity / float(epsilon)

    def gamma_k(t):
        return 0.5 / (1.0 + 0.1 * np.sqrt(t + 1.0))

    def alpha_k(t):
        return 0.001 / (1.0 + 0.01 * t)

    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}
    for t in range(n_iters):
        noisy = alphas + rng.laplace(0.0, b_scale, size=(n_agents, m))
        new_alphas = alphas.copy()
        for i, agent in enumerate(agents):
            grad_i = local_gradient(alphas[i], agent)
            diff_sum = np.zeros(m, dtype=float)
            for j in range(n_agents):
                if i != j and W[i, j] != 0:
                    diff_sum += W[i, j] * (noisy[j] - alphas[i])
            new_alphas[i] = alphas[i] + gamma_k(t) * diff_sum - alpha_k(t) * grad_i
        alphas = new_alphas
        _record_opt_gap(history, alphas, alpha_star)

    history['alphas'] = alphas
    history['alpha_mean'] = np.mean(alphas, axis=0)
    return history


# def run_push_sum_dgd(W, n_iters, alpha_star, agents, step, seed=0):
#     """Bonus : DGD Push-Sum orienté."""
#     rng = np.random.default_rng(seed)
#     W = np.asarray(W, dtype=float)
#     n_agents = len(agents)
#     m = alpha_star.size
#     x = rng.normal(0.0, 1e-3, size=(n_agents, m))
#     phi = np.ones((n_agents, 1), dtype=float)
#     history = {'mean_gap': []}

#     for _ in range(n_iters):
#         x = W @ x
#         phi = np.clip(W @ phi, 1e-12, None)
#         z = x / phi
#         grad = np.vstack([local_gradient(z[i], agents[i]) for i in range(n_agents)])
#         x = x - step * grad
#         z_mean = np.mean(x / phi, axis=0)
#         history['mean_gap'].append(float(np.linalg.norm(z_mean - alpha_star)))

#     history['alpha_mean'] = np.mean(x / phi, axis=0)
#     return history

def run_push_sum_dgd_directed(agents, P_col, alpha_star, step, n_iters, x0=None, seed=0, random_init=False):
    rng = np.random.default_rng(seed)
    P_col = np.asarray(P_col, dtype=float)
    n_agents = len(agents)
    m = alpha_star.size

    # Initialisation
    if x0 is not None:
        X = np.tile(np.asarray(x0).reshape(-1), (n_agents, 1))
    elif random_init:
        X = rng.normal(0.0, 1e-3, size=(n_agents, m))
    else:
        X = np.zeros((n_agents, m), dtype=float)

    w = np.ones(n_agents, dtype=float)
    # Z est la variable de décision réelle (X divisé par le poids w)
    Z = X / w[:, None]

    # Utilisation du même dictionnaire history que les autres
    history = {'mean_gap': [], 'max_gap': [], 'consensus_gap': []}

    for _ in range(n_iters):
        # Calcul du gradient sur la variable corrigée Z
        G = np.vstack([local_gradient(Z[i], agents[i]) for i in range(n_agents)])
        
        # Mise à jour Push-Sum
        X = P_col @ X - step * G
        w = np.clip(P_col @ w, 1e-12, None) # Éviter la division par zéro
        
        # Mise à jour de la variable locale
        Z = X / w[:, None]
        
        # C'EST ICI QUE ÇA CHANGE : on enregistre le Max Gap
        _record_opt_gap(history, Z, alpha_star)

    history['alphas'] = Z
    history['alpha_mean'] = np.mean(Z, axis=0)
    return history