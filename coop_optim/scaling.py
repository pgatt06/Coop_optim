import time

import numpy as np

from .algorithms import run_dgd, run_gradient_tracking
from .distributed_objectives import (
    aggregate_gradient_norm,
    centralized_solution_from_agents,
    make_streaming_agent_data,
)
from .graphs import make_complete_adjacency, metropolis_weights, spectral_beta


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
    except Exception as exc:  # pragma: no cover - defensive guard for long searches
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
