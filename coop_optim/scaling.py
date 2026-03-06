import time
import numpy as np

from .algorithms import run_dgd, run_gradient_tracking
from .graphs import complete_graph_weights
from .kernel_utils import cross_cov_matrix, cov_matrix


def first_index_below(curve, threshold):
    curve = np.asarray(curve, dtype=float)
    idx = np.where(curve <= threshold)[0]
    return int(idx[0]) if idx.size > 0 else -1


def build_agents_streaming(x_data, y_data, x_landmarks, n_agents, sigma=0.5, nu=1.0, batch_size=4096):
    """
    Construit les modèles quadratiques locaux sans stocker toute la matrice Knm.
    """
    x_data = np.asarray(x_data, dtype=float).reshape(-1)
    y_data = np.asarray(y_data, dtype=float).reshape(-1)
    x_landmarks = np.asarray(x_landmarks, dtype=float).reshape(-1)
    m = x_landmarks.size

    Kmm = cov_matrix(x_landmarks)
    I_m = np.eye(m)
    idx_split = np.array_split(np.arange(x_data.size), n_agents)

    agents = []
    for idx in idx_split:
        Q_i = (sigma ** 2 / n_agents) * Kmm + (nu / n_agents) * I_m
        b_i = np.zeros(m, dtype=float)

        for start in range(0, idx.size, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = x_data[batch_idx]
            yb = y_data[batch_idx]
            Mb = cross_cov_matrix(xb, x_landmarks)
            Q_i += Mb.T @ Mb
            b_i += Mb.T @ yb

        agents.append({'indices': idx, 'Q': Q_i, 'b': b_i})
    return agents


def centralized_alpha_from_agents(agents):
    Q_tot = np.zeros_like(agents[0]['Q'])
    b_tot = np.zeros_like(agents[0]['b'])
    for ag in agents:
        Q_tot += ag['Q']
        b_tot += ag['b']
    return np.linalg.solve(Q_tot, b_tot)


def compute_model_for_n(
    x_all,
    y_all,
    n_cur,
    n_agents,
    sigma=0.5,
    nu=1.0,
    seed_landmarks=0,
    batch_size=4096,
):
    m_cur = int(np.ceil(np.sqrt(n_cur)))
    x_n = np.asarray(x_all, dtype=float).reshape(-1)[:n_cur]
    y_n = np.asarray(y_all, dtype=float).reshape(-1)[:n_cur]

    rng = np.random.default_rng(seed_landmarks + n_cur)
    ind = np.sort(rng.choice(np.arange(n_cur), size=m_cur, replace=False))
    x_sel = x_n[ind]

    agents = build_agents_streaming(
        x_n,
        y_n,
        x_sel,
        n_agents=n_agents,
        sigma=sigma,
        nu=nu,
        batch_size=batch_size,
    )
    alpha_star = centralized_alpha_from_agents(agents)
    Lmax = max(float(np.linalg.eigvalsh(ag['Q'])[-1]) for ag in agents)
    return {
        'n': int(n_cur),
        'm': int(m_cur),
        'x_sel': x_sel,
        'alpha_star': alpha_star,
        'agents': agents,
        'Lmax': float(Lmax),
    }


def evaluate_n(
    x_all,
    y_all,
    n_cur,
    *,
    n_agents,
    sigma,
    nu,
    threshold,
    T_probe,
    batch_size,
    per_eval_time_limit,
    max_growth,
    progress_ratio,
    seed,
):
    """
    Évalue si un n est faisable selon stabilité + progrès + budget temps.
    """
    t0 = time.perf_counter()
    info = {'n': int(n_cur), 'm': int(np.ceil(np.sqrt(n_cur)))}

    try:
        model = compute_model_for_n(
            x_all,
            y_all,
            n_cur,
            n_agents=n_agents,
            sigma=sigma,
            nu=nu,
            seed_landmarks=10,
            batch_size=batch_size,
        )
        agents = model['agents']
        alpha_star = model['alpha_star']
        Lmax = model['Lmax']
        m_cur = model['m']

        T_eff = max(20, min(T_probe, int(12000 / max(1, m_cur))))
        W = complete_graph_weights(n_agents)
        eta_dgd = 0.9 / Lmax
        eta_gt = 0.2 / Lmax

        hist_dgd = run_dgd(agents, W, alpha_star, step=eta_dgd, n_iters=T_eff, seed=seed, random_init=False)
        hist_gt = run_gradient_tracking(agents, W, alpha_star, step=eta_gt, n_iters=T_eff, seed=seed, random_init=False)
        mean_dgd = np.asarray(hist_dgd['mean_gap'], dtype=float)
        mean_gt = np.asarray(hist_gt['mean_gap'], dtype=float)

        finite = bool(np.isfinite(mean_dgd).all() and np.isfinite(mean_gt).all())
        dgd_ratio = float(mean_dgd[-1] / max(mean_dgd[0], 1e-16))
        gt_ratio = float(mean_gt[-1] / max(mean_gt[0], 1e-16))
        stable = bool((dgd_ratio <= max_growth) and (gt_ratio <= max_growth))
        progress = bool((dgd_ratio <= progress_ratio) or (gt_ratio <= progress_ratio))
        elapsed = float(time.perf_counter() - t0)
        feasible = bool(finite and stable and progress and (elapsed <= per_eval_time_limit))

        reason = 'ok'
        if not finite:
            reason = 'nan_or_inf'
        elif not stable:
            reason = 'unstable'
        elif not progress:
            reason = 'no_progress'
        elif elapsed > per_eval_time_limit:
            reason = 'too_slow'

        info.update({
            'T_eff': int(T_eff),
            'elapsed_s': float(elapsed),
            'dgd_initial': float(mean_dgd[0]),
            'dgd_final': float(mean_dgd[-1]),
            'gt_initial': float(mean_gt[0]),
            'gt_final': float(mean_gt[-1]),
            'dgd_ratio': float(dgd_ratio),
            'gt_ratio': float(gt_ratio),
            'dgd_it': int(first_index_below(mean_dgd, threshold)),
            'gt_it': int(first_index_below(mean_gt, threshold)),
            'feasible': bool(feasible),
            'reason': reason,
        })

        compact_model = {'x_sel': model['x_sel'], 'alpha_star': model['alpha_star']}
        return feasible, info, compact_model
    except MemoryError:
        info.update({'elapsed_s': float(time.perf_counter() - t0), 'feasible': False, 'reason': 'memory_error'})
        return False, info, None
    except np.linalg.LinAlgError:
        info.update({'elapsed_s': float(time.perf_counter() - t0), 'feasible': False, 'reason': 'linear_algebra_error'})
        return False, info, None
    except Exception as exc:
        info.update({'elapsed_s': float(time.perf_counter() - t0), 'feasible': False, 'reason': f'error:{type(exc).__name__}'})
        return False, info, None


def find_largest_n_possible(
    x_all,
    y_all,
    *,
    n_min,
    n_max,
    n_agents,
    sigma=0.5,
    nu=1.0,
    threshold=1.0,
    T_probe=120,
    growth=2.0,
    max_evals=16,
    batch_size=4096,
    per_eval_time_limit=120.0,
    total_time_budget=1800.0,
    max_growth=10.0,
    progress_ratio=0.95,
    seed=0,
):
    """
    Recherche du plus grand n faisable :
    1) croissance exponentielle pour trouver une borne d'échec,
    2) dichotomie entre dernier succès et premier échec.
    """
    t_global = time.perf_counter()
    logs = []
    model_cache = {}
    eval_count = 0

    best_n = None
    first_fail_n = None
    n_cur = int(n_min)

    while eval_count < max_evals and n_cur <= n_max:
        if (time.perf_counter() - t_global) > total_time_budget:
            break
        feasible, info, model = evaluate_n(
            x_all,
            y_all,
            n_cur,
            n_agents=n_agents,
            sigma=sigma,
            nu=nu,
            threshold=threshold,
            T_probe=T_probe,
            batch_size=batch_size,
            per_eval_time_limit=per_eval_time_limit,
            max_growth=max_growth,
            progress_ratio=progress_ratio,
            seed=seed,
        )
        logs.append(info)
        eval_count += 1

        if feasible:
            best_n = n_cur
            if model is not None:
                model_cache[n_cur] = model
            if n_cur == n_max:
                return best_n, logs, model_cache
            n_next = int(max(n_cur + 1, np.floor(n_cur * growth)))
            n_cur = min(n_next, n_max)
        else:
            first_fail_n = n_cur
            break

    if best_n is None:
        raise RuntimeError('Aucune valeur faisable trouvée à partir de n_min.')
    if first_fail_n is None:
        return best_n, logs, model_cache

    lo, hi = int(best_n), int(first_fail_n)
    while eval_count < max_evals and (hi - lo) > 1:
        if (time.perf_counter() - t_global) > total_time_budget:
            break
        mid = (lo + hi) // 2
        feasible, info, model = evaluate_n(
            x_all,
            y_all,
            mid,
            n_agents=n_agents,
            sigma=sigma,
            nu=nu,
            threshold=threshold,
            T_probe=T_probe,
            batch_size=batch_size,
            per_eval_time_limit=per_eval_time_limit,
            max_growth=max_growth,
            progress_ratio=progress_ratio,
            seed=seed,
        )
        logs.append(info)
        eval_count += 1
        if feasible:
            lo = mid
            best_n = mid
            if model is not None:
                model_cache[mid] = model
        else:
            hi = mid
    return int(best_n), logs, model_cache
