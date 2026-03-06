#!/usr/bin/env python3
"""
Script simple pour trouver le plus grand n faisable (Partie I).

Idée:
- on impose m = ceil(sqrt(n))
- on construit les matrices locales en streaming (pour éviter d'exploser la RAM)
- on teste DGD + GT rapidement
- on fait une recherche exponentielle puis dichotomique sur n
"""

import argparse
import csv
import math
import os
import pickle
import time
import warnings
from pathlib import Path
from tempfile import gettempdir

import numpy as np

# Matplotlib en mode non interactif (important en terminal/sandbox)
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "mplconfig_codex"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------
# Configuration par défaut
# -------------------------------
# Chemin du dataset principal (Partie I).
DATA_PATH = Path("data/first_database.pkl")
# Dossier de sortie des logs/figures (ici: dossier courant).
OUT_DIR = Path("./tools/largest_n_possible_output")

# n de départ pour la recherche.
N_MIN = 100
# n maximum testé. Mettre 0 pour utiliser toute la base.
N_MAX = 0
# Nombre d'agents.
A = 5
# Paramètre sigma du problème ridge noyau.
SIGMA = 0.5
# Régularisation forte convexité (nu).
NU = 1.0

# Seuil de gap pour la métrique « itérations pour atteindre le seuil ».
THRESHOLD = 1.0
# Nombre max d'itérations de probe par évaluation de n.
T_PROBE = 300
# Facteur de croissance en phase exponentielle (ex: 2 => doublement).
GROWTH = 2.0
# Nombre total max d'évaluations de n (exp + dichotomie).
MAX_EVALS = 16
# Taille de batch pour construire H_i et b_i en streaming.
BATCH_SIZE = 4096
# Temps max autorisé pour une seule évaluation de n (secondes).
PER_EVAL_TIME_LIMIT = 120.0
# Budget temps global pour toute la recherche (secondes).
TOTAL_TIME_BUDGET = 1800.0
# Critère de stabilité: gap_final / gap_initial doit rester <= MAX_GROWTH.
MAX_GROWTH = 10.0
# Critère de progrès: au moins une méthode doit finir <= PROGRESS_RATIO * initial.
PROGRESS_RATIO = 0.85
# Graine aléatoire globale (initialisations DGD/GT + reproductibilité).
SEED = 0


# -------------------------------
# Outils de base
# -------------------------------
def load_first_database(path):
    """Charge first_database.pkl et retourne x, y en vecteurs 1D float."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        with path.open("rb") as f:
            x, y = pickle.load(f)

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x et y n'ont pas la même taille: {x.shape[0]} vs {y.shape[0]}")

    return x, y


def kernel_matrix(x_points, x_landmarks):
    """Matrice noyau RBF: exp(-(x-z)^2)."""
    x = np.asarray(x_points, dtype=float).reshape(-1, 1)
    z = np.asarray(x_landmarks, dtype=float).reshape(1, -1)
    return np.exp(-((x - z) ** 2))


# -------------------------------
# Construction du problème
# -------------------------------
def build_agents_streaming(x_data, y_data, x_landmarks, a, sigma, nu, batch_size):
    """
    Construit pour chaque agent:
    - H_i (hessienne locale quadratique)
    - b_i
    sans stocker toute la matrice Knm en mémoire.
    """
    m = len(x_landmarks)
    kmm = kernel_matrix(x_landmarks, x_landmarks)
    eye_m = np.eye(m)

    idx_split = np.array_split(np.arange(len(x_data)), a)
    agents = []

    for i, idx in enumerate(idx_split):
        h_i = (sigma**2 / a) * kmm + (nu / a) * eye_m
        b_i = np.zeros(m, dtype=float)

        for start in range(0, len(idx), batch_size):
            idb = idx[start : start + batch_size]
            xb = x_data[idb]
            yb = y_data[idb]
            kb = kernel_matrix(xb, x_landmarks)

            h_i += kb.T @ kb
            b_i += kb.T @ yb

        eigvals = np.linalg.eigvalsh(h_i)
        agents.append(
            {
                "id": i,
                "H": h_i,
                "b": b_i,
                "L": float(eigvals[-1]),
                "mu": float(eigvals[0]),
            }
        )

    return agents


def centralized_alpha_from_agents(agents):
    """Recompose le problème global et résout alpha* (centralisé)."""
    h_tot = np.zeros_like(agents[0]["H"])
    b_tot = np.zeros_like(agents[0]["b"])

    for ag in agents:
        h_tot += ag["H"]
        b_tot += ag["b"]

    return np.linalg.solve(h_tot, b_tot)


def compute_model_for_n(x_all, y_all, n_cur, a, sigma, nu, batch_size, seed_landmarks=10):
    """
    Pour une valeur n:
    - m = ceil(sqrt(n))
    - sélection des landmarks
    - construction agents
    - alpha* centralisé
    """
    m_cur = int(math.ceil(math.sqrt(n_cur)))

    x_n = x_all[:n_cur].copy()
    y_n = y_all[:n_cur].copy()

    rng = np.random.default_rng(seed_landmarks + n_cur)
    ind = np.sort(rng.choice(np.arange(n_cur), size=m_cur, replace=False))
    x_sel = x_n[ind]

    agents = build_agents_streaming(
        x_n,
        y_n,
        x_sel,
        a=a,
        sigma=sigma,
        nu=nu,
        batch_size=batch_size,
    )

    alpha_star = centralized_alpha_from_agents(agents)
    lmax = max(ag["L"] for ag in agents)

    return x_sel, alpha_star, agents, float(lmax), m_cur


# -------------------------------
# Algorithmes distribués (version simple)
# -------------------------------
def grad_all(agents, a_mat):
    """Gradient local sur chaque agent."""
    g = np.zeros_like(a_mat)
    for i, ag in enumerate(agents):
        g[i] = ag["H"] @ a_mat[i] - ag["b"]
    return g


def mean_gap(a_mat, alpha_star):
    """Gap moyen ||alpha_i - alpha*|| sur les agents."""
    return float(np.mean(np.linalg.norm(a_mat - alpha_star.reshape(1, -1), axis=1)))


def run_dgd(agents, w, alpha_star, t_iters, eta, seed):
    rng = np.random.default_rng(seed)
    a = len(agents)
    m = agents[0]["H"].shape[0]

    a_mat = rng.normal(0.0, 0.1, size=(a, m))
    curve = np.zeros(t_iters + 1, dtype=float)
    curve[0] = mean_gap(a_mat, alpha_star)

    for t in range(t_iters):
        g = grad_all(agents, a_mat)
        a_mat = w @ a_mat - eta * g
        curve[t + 1] = mean_gap(a_mat, alpha_star)

    return curve


def run_gradient_tracking(agents, w, alpha_star, t_iters, eta, seed):
    rng = np.random.default_rng(seed)
    a = len(agents)
    m = agents[0]["H"].shape[0]

    a_mat = rng.normal(0.0, 0.1, size=(a, m))
    g = grad_all(agents, a_mat)
    s = g.copy()

    curve = np.zeros(t_iters + 1, dtype=float)
    curve[0] = mean_gap(a_mat, alpha_star)

    for t in range(t_iters):
        a_next = w @ a_mat - eta * s
        g_next = grad_all(agents, a_next)
        s = w @ s + (g_next - g)
        a_mat, g = a_next, g_next
        curve[t + 1] = mean_gap(a_mat, alpha_star)

    return curve


# -------------------------------
# Évaluation d'un n
# -------------------------------
def first_index_below(curve, threshold):
    idx = np.where(curve <= threshold)[0]
    return int(idx[0]) if len(idx) > 0 else -1


def evaluate_n(
    x_all,
    y_all,
    n_cur,
    *,
    a,
    sigma,
    nu,
    threshold,
    t_probe,
    batch_size,
    per_eval_time_limit,
    max_growth,
    progress_ratio,
    seed,
):
    t0 = time.perf_counter()

    info = {
        "n": int(n_cur),
        "m": int(math.ceil(math.sqrt(n_cur))),
    }

    try:
        x_sel, alpha_star, agents, lmax, m_cur = compute_model_for_n(
            x_all,
            y_all,
            n_cur,
            a=a,
            sigma=sigma,
            nu=nu,
            batch_size=batch_size,
            seed_landmarks=10,
        )

        # Moins d'itérations quand m grossit
        t_eff = max(20, min(t_probe, int(12000 / max(1, m_cur))))

        # Graphe complet (mélange parfait) pour le probe
        w_complete = np.full((a, a), 1.0 / a, dtype=float)

        eta_dgd = 0.9 / lmax
        eta_gt = 0.2 / lmax

        curve_dgd = run_dgd(agents, w_complete, alpha_star, t_eff, eta_dgd, seed)
        curve_gt = run_gradient_tracking(agents, w_complete, alpha_star, t_eff, eta_gt, seed)

        finite = np.isfinite(curve_dgd).all() and np.isfinite(curve_gt).all()

        dgd_ratio = float(curve_dgd[-1] / max(curve_dgd[0], 1e-16))
        gt_ratio = float(curve_gt[-1] / max(curve_gt[0], 1e-16))

        stable = (dgd_ratio <= max_growth) and (gt_ratio <= max_growth)
        progress = (dgd_ratio <= progress_ratio) or (gt_ratio <= progress_ratio)

        elapsed = float(time.perf_counter() - t0)
        feasible = bool(finite and stable and progress and (elapsed <= per_eval_time_limit))

        if not finite:
            reason = "nan_or_inf"
        elif not stable:
            reason = "unstable"
        elif not progress:
            reason = "no_progress"
        elif elapsed > per_eval_time_limit:
            reason = "too_slow"
        else:
            reason = "ok"

        info.update(
            {
                "T_eff": int(t_eff),
                "elapsed_s": elapsed,
                "dgd_initial": float(curve_dgd[0]),
                "dgd_final": float(curve_dgd[-1]),
                "gt_initial": float(curve_gt[0]),
                "gt_final": float(curve_gt[-1]),
                "dgd_ratio": dgd_ratio,
                "gt_ratio": gt_ratio,
                "dgd_it": first_index_below(curve_dgd, threshold),
                "gt_it": first_index_below(curve_gt, threshold),
                "feasible": feasible,
                "reason": reason,
            }
        )

        model = {
            "x_sel": x_sel,
            "alpha_star": alpha_star,
        }
        return feasible, info, model

    except MemoryError:
        info.update({"elapsed_s": float(time.perf_counter() - t0), "feasible": False, "reason": "memory_error"})
        return False, info, None
    except np.linalg.LinAlgError:
        info.update({"elapsed_s": float(time.perf_counter() - t0), "feasible": False, "reason": "linear_algebra_error"})
        return False, info, None
    except Exception as e:
        info.update({"elapsed_s": float(time.perf_counter() - t0), "feasible": False, "reason": f"error:{type(e).__name__}"})
        return False, info, None


# -------------------------------
# Recherche du plus grand n
# -------------------------------
def find_largest_n_possible(
    x_all,
    y_all,
    *,
    n_min,
    n_max,
    a,
    sigma,
    nu,
    threshold,
    t_probe,
    growth,
    max_evals,
    batch_size,
    per_eval_time_limit,
    total_time_budget,
    max_growth,
    progress_ratio,
    seed,
):
    logs = []
    model_cache = {}

    eval_count = 0
    best_n = None
    first_fail_n = None

    t_global = time.perf_counter()
    n_cur = int(n_min)

    # 1) Phase exponentielle
    while eval_count < max_evals and n_cur <= n_max:
        if (time.perf_counter() - t_global) > total_time_budget:
            break

        feasible, info, model = evaluate_n(
            x_all,
            y_all,
            n_cur,
            a=a,
            sigma=sigma,
            nu=nu,
            threshold=threshold,
            t_probe=t_probe,
            batch_size=batch_size,
            per_eval_time_limit=per_eval_time_limit,
            max_growth=max_growth,
            progress_ratio=progress_ratio,
            seed=seed,
        )

        eval_count += 1
        logs.append(info)
        print(
            f"[eval {eval_count:02d}] n={info['n']}, m={info['m']}, "
            f"feasible={info['feasible']}, reason={info['reason']}, "
            f"elapsed={float(info['elapsed_s']):.2f}s"
        )

        if feasible:
            best_n = n_cur
            if model is not None:
                model_cache[n_cur] = model

            if n_cur == n_max:
                return best_n, logs, model_cache

            n_next = int(max(n_cur + 1, math.floor(n_cur * growth)))
            n_cur = min(n_next, n_max)
        else:
            first_fail_n = n_cur
            break

    if best_n is None:
        raise RuntimeError("Aucun n faisable trouvé depuis n_min")

    if first_fail_n is None:
        return best_n, logs, model_cache

    # 2) Phase dichotomique
    lo, hi = best_n, first_fail_n
    while eval_count < max_evals and (hi - lo) > 1:
        if (time.perf_counter() - t_global) > total_time_budget:
            break

        mid = (lo + hi) // 2

        feasible, info, model = evaluate_n(
            x_all,
            y_all,
            mid,
            a=a,
            sigma=sigma,
            nu=nu,
            threshold=threshold,
            t_probe=t_probe,
            batch_size=batch_size,
            per_eval_time_limit=per_eval_time_limit,
            max_growth=max_growth,
            progress_ratio=progress_ratio,
            seed=seed,
        )

        eval_count += 1
        logs.append(info)
        print(
            f"[eval {eval_count:02d}] n={info['n']}, m={info['m']}, "
            f"feasible={info['feasible']}, reason={info['reason']}, "
            f"elapsed={float(info['elapsed_s']):.2f}s"
        )

        if feasible:
            lo = mid
            best_n = mid
            if model is not None:
                model_cache[mid] = model
        else:
            hi = mid

    return best_n, logs, model_cache


# -------------------------------
# Sauvegardes résultats
# -------------------------------
def save_logs_csv(path, logs):
    if not logs:
        return

    fieldnames = sorted({k for row in logs for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)


def plot_logs(out_dir, logs, threshold):
    feasible_records = sorted([r for r in logs if bool(r.get("feasible", False))], key=lambda r: int(r["n"]))
    if not feasible_records:
        return

    ns = [int(r["n"]) for r in feasible_records]
    dgd_final = [float(r["dgd_final"]) for r in feasible_records]
    gt_final = [float(r["gt_final"]) for r in feasible_records]
    dgd_it = [float(r["dgd_it"]) if int(r["dgd_it"]) >= 0 else np.nan for r in feasible_records]
    gt_it = [float(r["gt_it"]) if int(r["gt_it"]) >= 0 else np.nan for r in feasible_records]

    plt.figure()
    plt.loglog(ns, dgd_final, marker="o", label="DGD final gap")
    plt.loglog(ns, gt_final, marker="o", label="GT final gap")
    plt.xlabel("n")
    plt.ylabel("final mean optimality gap")
    plt.title("Probe convergence quality vs n")
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "n_scaling_final_gap.pdf", format="pdf")
    plt.close()

    plt.figure()
    plt.plot(ns, dgd_it, marker="o", label="DGD iters to threshold")
    plt.plot(ns, gt_it, marker="o", label="GT iters to threshold")
    plt.xlabel("n")
    plt.ylabel(f"iterations to mean gap <= {threshold}")
    plt.title("Probe speed vs n")
    plt.grid(True, ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "n_scaling_iterations.pdf", format="pdf")
    plt.close()


# -------------------------------
# Programme principal
# -------------------------------
def parse_args():
    # Analyseur d'arguments minimal : tu peux tout changer dans les constantes du haut.
    parser = argparse.ArgumentParser(description="Trouver le plus grand n faisable (Partie I)")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Chemin du fichier first_database.pkl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Dossier de sortie des logs/figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all = load_first_database(args.data)
    n_dataset = len(x_all)

    n_max = n_dataset if N_MAX <= 0 else min(N_MAX, n_dataset)
    n_min = max(2, N_MIN)

    if n_min > n_max:
        raise ValueError(f"Bornes invalides: n_min={n_min} > n_max={n_max}")

    best_n, logs, model_cache = find_largest_n_possible(
        x_all,
        y_all,
        n_min=n_min,
        n_max=n_max,
        a=A,
        sigma=SIGMA,
        nu=NU,
        threshold=THRESHOLD,
        t_probe=T_PROBE,
        growth=GROWTH,
        max_evals=MAX_EVALS,
        batch_size=BATCH_SIZE,
        per_eval_time_limit=PER_EVAL_TIME_LIMIT,
        total_time_budget=TOTAL_TIME_BUDGET,
        max_growth=MAX_GROWTH,
        progress_ratio=PROGRESS_RATIO,
        seed=SEED,
    )

    logs_csv = args.out_dir / "largest_n_search_logs.csv"
    save_logs_csv(logs_csv, logs)
    plot_logs(args.out_dir, logs, threshold=THRESHOLD)

    print()
    print(f"Dataset size: {n_dataset}")
    print(f"Largest feasible n found: {best_n}")
    print(f"Logs saved to: {logs_csv}")

    if best_n in model_cache:
        model = model_cache[best_n]
        np.savez(
            args.out_dir / "best_model.npz",
            n=np.array([best_n], dtype=int),
            x_sel=model["x_sel"],
            alpha_star=model["alpha_star"],
        )
        print(f"Best model saved to: {args.out_dir / 'best_model.npz'}")


if __name__ == "__main__":
    main()
