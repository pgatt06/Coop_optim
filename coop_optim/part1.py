import numpy as np

from .algorithms import (
    dual_lipschitz_constant,
    run_async_dgd,
    run_consensus_admm,
    run_dgd,
    run_dgd_packet_loss,
    run_dual_decomposition,
    run_gradient_tracking,
    run_push_sum_dgd,
)
from .centralized import build_nystrom_problem, objective, smoothness_and_strong_convexity, solve_centralized
from .config import (
    DATA_DIR,
    FIGURES_DIR,
    GRAPH_SPECS,
    M_PART1,
    N_AGENTS,
    N_PART1,
    NU,
    PART1_ACTIVE_PROB,
    PART1_ADMM_ITERS,
    PART1_BREAK_ITERS,
    PART1_DD_ITERS,
    PART1_DGD_ITERS,
    PART1_GT_ITERS,
    PART1_PACKET_LOSS,
    PART1_SCALING_BATCH_SIZE,
    PART1_SCALING_ETA_MULTIPLIERS,
    PART1_SCALING_LONG_ITER_BUDGET,
    PART1_SCALING_MAX_AGENTS,
    PART1_SCALING_MAX_EVALS,
    PART1_SCALING_MIN_AGENTS,
    PART1_SCALING_PER_EVAL_LIMIT_S,
    PART1_SCALING_PROBE_ITERS,
    PART1_SCALING_TARGET_LOCAL_SIZE,
    PART1_SCALING_THRESHOLD,
    PART1_SCALING_TOTAL_BUDGET_S,
    SEED,
    SIGMA,
)
from .data_utils import ensure_dir, load_first_database
from .distributed_objectives import make_agent_data
from .graphs import (
    build_push_sum_column_matrix,
    get_graph,
    incidence_matrix,
    make_complete_adjacency,
    make_directed_cycle_weights,
    metropolis_weights,
    spectral_beta,
)
from .kernel_utils import predict
from .plotting import (
    save_agent_gap_grid_plot,
    save_dataset_plot,
    save_history_plot,
    save_reconstruction_plot,
    save_xy_plot,
)
from .scaling import compute_model_for_n, find_largest_n_possible, select_scaling_plot_ns, tune_best_histories


def part1_steps(agents, W, incidence):
    """Compute theory-motivated stepsizes and penalty parameters for Part I."""
    L_max = max(agent["L"] for agent in agents)
    mu_min = min(agent["mu"] for agent in agents)
    beta = spectral_beta(W)
    L_dual = dual_lipschitz_constant(agents, incidence)
    return {
        "L_max": float(L_max),
        "mu_min": float(mu_min),
        "beta": float(beta),
        "L_dual": float(L_dual),
        "eta_dgd": float(min(0.9 * (1.0 - beta) / L_max, 0.9 / L_max)),
        "eta_gt": float(0.2 / L_max),
        "tau_dual": float(1.0 / (2.0 * L_dual)),
        "rho_admm": float((mu_min * L_max) ** 0.5),
    }


def write_part1_summary(path, records, line_steps, scaling_records, n_best):
    """Write a compact theory-and-checklist summary alongside the figures."""
    lines = [
        "Part I checklist",
        "================",
        "",
        "1. Baseline distributed methods use the course assumptions: connected undirected graph and doubly-stochastic mixing.",
        f"   DGD step uses min(0.9*(1-beta)/L, 0.9/L) with beta={line_steps['beta']:.4f} and L_max={line_steps['L_max']:.4f}.",
        f"   GT step uses 0.2/L_max = {line_steps['eta_gt']:.3e}.",
        f"   Dual decomposition uses tau = 1/(2*L_dual) = {line_steps['tau_dual']:.3e}.",
        f"   ADMM uses rho = sqrt(mu_min*L_max) = {line_steps['rho_admm']:.3e}.",
        "",
        "2. Directed communication breaks the DGD proof because the average iterate is no longer preserved when the matrix is not doubly stochastic.",
        "   Packet losses create a time-varying graph that may disconnect the network and violate the fixed-mixing assumption.",
        "   Asynchrony violates the synchronous update model used in the standard contraction argument.",
        "",
        "3. Push-sum restores the missing Perron normalization on directed graphs by tracking weights w_i^t and using z_i^t = x_i^t / w_i^t.",
        "",
        "4. Large-n scaling keeps m = ceil(sqrt(n)).",
        "   The scaling suite also logs consensus gaps and aggregate gradient norms, so it still gives convergence indicators even if a reference solution becomes unavailable.",
        f"   Largest feasible n found under the current compute budget: {n_best}.",
        "",
        "Generated figures",
        "-----------------",
    ]
    for name in records:
        lines.append(f"- {name}")
    if scaling_records:
        lines.append("")
        lines.append("Feasible scaling records")
        lines.append("------------------------")
        for rec in scaling_records:
            lines.append(
                f"n={rec['n']}, m={rec['m']}, agents={rec['n_agents']}, "
                f"dgd_final={rec['dgd_final']:.3e}, gt_final={rec['gt_final']:.3e}"
            )

    ensure_dir(str(path.parent))
    with open(path, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")


def run_line_baselines(problem, agents, alpha_star):
    """Run the four baseline methods on the line graph and save core figures."""
    adj_line, W_line = get_graph("line", N_AGENTS, seed=SEED)
    incidence_line, _ = incidence_matrix(adj_line)
    line_steps = part1_steps(agents, W_line, incidence_line)

    histories = {
        "DGD": run_dgd(agents, W_line, alpha_star, step=line_steps["eta_dgd"], n_iters=PART1_DGD_ITERS, seed=SEED),
        "Gradient tracking": run_gradient_tracking(
            agents,
            W_line,
            alpha_star,
            step=line_steps["eta_gt"],
            n_iters=PART1_GT_ITERS,
            seed=SEED,
        ),
        "Dual decomposition": run_dual_decomposition(
            agents,
            incidence_line,
            alpha_star,
            step=line_steps["tau_dual"],
            n_iters=PART1_DD_ITERS,
        ),
        "ADMM": run_consensus_admm(
            agents,
            adj_line,
            alpha_star,
            rho=line_steps["rho_admm"],
            n_iters=PART1_ADMM_ITERS,
        ),
    }

    save_agent_gap_grid_plot(
        {label: hist["agent_gaps"] for label, hist in histories.items()},
        FIGURES_DIR / "part1_gap_line.pdf",
        ylabel=r"$\|\alpha_i^t-\alpha^\star\|$",
        title="Part I - line graph",
    )

    x_query = np.linspace(-1.0, 1.0, 250)
    alpha_methods = {"Centralized": alpha_star}
    for label, hist in histories.items():
        alpha_methods[label] = hist["alpha_mean"]
    save_reconstruction_plot(
        alpha_methods,
        problem["x_n"],
        problem["y_n"],
        x_query,
        problem["x_m"],
        FIGURES_DIR / "part1_reconstruction_compare.pdf",
        title="Centralized vs distributed reconstruction",
    )
    return histories, adj_line, W_line, incidence_line, line_steps


def run_graph_sweep(agents, alpha_star):
    """Compare the baseline methods across the graph families from the subject."""
    histories_by_method = {
        "DGD": {},
        "Gradient tracking": {},
        "Dual decomposition": {},
        "ADMM": {},
    }
    method_to_file = {
        "DGD": "part1_dgd_graph_compare.pdf",
        "Gradient tracking": "part1_gt_graph_compare.pdf",
        "Dual decomposition": "part1_dual_graph_compare.pdf",
        "ADMM": "part1_admm_graph_compare.pdf",
    }

    for graph_name, graph_label in GRAPH_SPECS:
        adj, W = get_graph(graph_name, N_AGENTS, seed=SEED)
        incidence, _ = incidence_matrix(adj)
        params = part1_steps(agents, W, incidence)

        histories_by_method["DGD"][graph_label] = run_dgd(
            agents,
            W,
            alpha_star,
            step=params["eta_dgd"],
            n_iters=PART1_DGD_ITERS,
            seed=SEED,
        )["bar_gap"]
        histories_by_method["Gradient tracking"][graph_label] = run_gradient_tracking(
            agents,
            W,
            alpha_star,
            step=params["eta_gt"],
            n_iters=PART1_GT_ITERS,
            seed=SEED,
        )["bar_gap"]
        histories_by_method["Dual decomposition"][graph_label] = run_dual_decomposition(
            agents,
            incidence,
            alpha_star,
            step=params["tau_dual"],
            n_iters=PART1_DD_ITERS,
        )["bar_gap"]
        histories_by_method["ADMM"][graph_label] = run_consensus_admm(
            agents,
            adj,
            alpha_star,
            rho=params["rho_admm"],
            n_iters=PART1_ADMM_ITERS,
        )["bar_gap"]
        print(
            f"  [{graph_label}] beta={params['beta']:.6f}, "
            f"eta_dgd={params['eta_dgd']:.3e}, eta_gt={params['eta_gt']:.3e}, "
            f"tau_dual={params['tau_dual']:.3e}, rho_admm={params['rho_admm']:.3e}"
        )

    for method, curves in histories_by_method.items():
        save_history_plot(
            curves,
            FIGURES_DIR / method_to_file[method],
            ylabel=r"$\|\bar{\alpha}^t-\alpha^\star\|$",
            title=f"{method} - graph effect on the averaged iterate",
        )
    return histories_by_method


def run_break_and_push_sum(agents, alpha_star, adj_line, W_line, line_steps):
    """Run the convergence-breaking experiments and the push-sum recovery test."""
    W_directed = make_directed_cycle_weights(len(agents))
    P_col = build_push_sum_column_matrix(W_directed)

    baseline = run_dgd(
        agents,
        W_line,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        seed=SEED + 1,
        random_init=True,
    )
    directed = run_dgd(
        agents,
        W_directed,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        seed=SEED + 1,
        random_init=True,
    )
    packet_loss = run_dgd_packet_loss(
        agents,
        adj_line,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        p_loss=PART1_PACKET_LOSS,
        seed=SEED + 1,
    )
    asynchronous = run_async_dgd(
        agents,
        W_line,
        alpha_star,
        step=line_steps["eta_dgd"],
        n_iters=PART1_BREAK_ITERS,
        p_active=PART1_ACTIVE_PROB,
        seed=SEED + 1,
    )

    save_history_plot(
        {
            "Undirected baseline": baseline["mean_gap"],
            "Directed": directed["mean_gap"],
            "Packet losses": packet_loss["mean_gap"],
            "Asynchronous": asynchronous["mean_gap"],
        },
        FIGURES_DIR / "part1_break_convergence.pdf",
        ylabel="Mean optimality gap",
        title="Breaking convergence scenarios",
    )

    eta_push_sum = min(0.45 / line_steps["L_max"], 0.95 / line_steps["L_max"])
    push_sum = run_push_sum_dgd(
        agents,
        P_col,
        alpha_star,
        step=eta_push_sum,
        n_iters=PART1_BREAK_ITERS,
        seed=SEED + 1,
    )
    save_history_plot(
        {
            "Directed DGD": directed["mean_gap"],
            "Push-sum DGD": push_sum["mean_gap"],
        },
        FIGURES_DIR / "part1_push_sum_recovery.pdf",
        ylabel="Mean optimality gap",
        title="Directed communication: push-sum recovery",
    )

    print(f"  directed columns sums = {np.round(W_directed.sum(axis=0), 4)}")
    print(f"  push-sum column sums = {np.round(P_col.sum(axis=0), 4)}")
    print(
        f"  break suite final gaps: baseline={baseline['mean_gap'][-1]:.3e}, "
        f"directed={directed['mean_gap'][-1]:.3e}, "
        f"loss={packet_loss['mean_gap'][-1]:.3e}, async={asynchronous['mean_gap'][-1]:.3e}, "
        f"push-sum={push_sum['mean_gap'][-1]:.3e}"
    )


def run_scaling_suite(x, y):
    """Run the large-n search and save surrogate convergence diagnostics."""
    threshold = PART1_SCALING_THRESHOLD
    n_best, search_logs, model_cache = find_largest_n_possible(
        x,
        y,
        n_min=100,
        n_max=len(x),
        sigma=SIGMA,
        nu=NU,
        threshold=threshold,
        T_probe=PART1_SCALING_PROBE_ITERS,
        growth=2.0,
        max_evals=PART1_SCALING_MAX_EVALS,
        batch_size=PART1_SCALING_BATCH_SIZE,
        per_eval_time_limit=PART1_SCALING_PER_EVAL_LIMIT_S,
        total_time_budget=PART1_SCALING_TOTAL_BUDGET_S,
        min_agents=PART1_SCALING_MIN_AGENTS,
        max_agents=PART1_SCALING_MAX_AGENTS,
        target_local_size=PART1_SCALING_TARGET_LOCAL_SIZE,
        seed=SEED,
    )
    records = sorted([record for record in search_logs if record["feasible"]], key=lambda record: record["n"])

    save_xy_plot(
        {
            "DGD final gap": ([record["n"] for record in records], [record["dgd_final"] for record in records]),
            "GT final gap": ([record["n"] for record in records], [record["gt_final"] for record in records]),
        },
        FIGURES_DIR / "part1_scaling_final_gap.pdf",
        ylabel="Final mean optimality gap",
        title="Convergence quality vs n",
        xlabel="n",
        xscale="log",
        yscale="log",
    )
    save_xy_plot(
        {
            "DGD iters to threshold": ([record["n"] for record in records], [record["dgd_it"] for record in records]),
            "GT iters to threshold": ([record["n"] for record in records], [record["gt_it"] for record in records]),
        },
        FIGURES_DIR / "part1_scaling_iterations.pdf",
        ylabel=f"Iterations to mean gap <= {threshold}",
        title="Speed vs n",
        xlabel="n",
        xscale="log",
        yscale="linear",
    )
    save_xy_plot(
        {
            "DGD final consensus gap": (
                [record["n"] for record in records],
                [record["dgd_consensus_final"] for record in records],
            ),
            "GT final consensus gap": (
                [record["n"] for record in records],
                [record["gt_consensus_final"] for record in records],
            ),
            "DGD final aggregate grad norm": (
                [record["n"] for record in records],
                [record["dgd_grad_norm_final"] for record in records],
            ),
            "GT final aggregate grad norm": (
                [record["n"] for record in records],
                [record["gt_grad_norm_final"] for record in records],
            ),
        },
        FIGURES_DIR / "part1_scaling_surrogates.pdf",
        ylabel="Surrogate convergence metrics",
        title="Consensus and stationarity vs n",
        xlabel="n",
        xscale="log",
        yscale="log",
    )

    selected_ns = select_scaling_plot_ns(records)
    x_query = np.linspace(-1.0, 1.0, 250)
    curves = {}
    for n_cur in selected_ns:
        model = model_cache.get(n_cur)
        if model is None:
            model = compute_model_for_n(
                x,
                y,
                n_cur,
                sigma=SIGMA,
                nu=NU,
                min_agents=PART1_SCALING_MIN_AGENTS,
                max_agents=PART1_SCALING_MAX_AGENTS,
                target_local_size=PART1_SCALING_TARGET_LOCAL_SIZE,
                seed_landmarks=10,
                batch_size=PART1_SCALING_BATCH_SIZE,
            )
        curves[f"n={n_cur}"] = predict(model["alpha_star"], x_query, model["x_landmarks"])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.4, 4.8))
    for label, values in curves.items():
        plt.plot(x_query, values, lw=1.8, label=label)
    plt.grid(True, alpha=0.35)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.title("Centralized reconstructed functions for selected n")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "part1_scaling_functions.pdf")
    plt.close()

    best_model = compute_model_for_n(
        x,
        y,
        n_best,
        sigma=SIGMA,
        nu=NU,
        min_agents=PART1_SCALING_MIN_AGENTS,
        max_agents=PART1_SCALING_MAX_AGENTS,
        target_local_size=PART1_SCALING_TARGET_LOCAL_SIZE,
        seed_landmarks=11,
        batch_size=max(PART1_SCALING_BATCH_SIZE, 4096),
    )
    W_best = metropolis_weights(make_complete_adjacency(best_model["n_agents"]))
    best_dgd, best_gt, T_long = tune_best_histories(
        best_model["agents"],
        best_model["alpha_star"],
        W_best,
        eta_multipliers=PART1_SCALING_ETA_MULTIPLIERS,
        iteration_budget=PART1_SCALING_LONG_ITER_BUDGET,
        seed=SEED + 2,
    )

    save_history_plot(
        {
            f"DGD tuned (eta={best_dgd['eta']:.2e})": best_dgd["history"]["mean_gap"],
            f"GT tuned (eta={best_gt['eta']:.2e})": best_gt["history"]["mean_gap"],
        },
        FIGURES_DIR / "part1_scaling_best_tuned.pdf",
        ylabel="Mean optimality gap",
        title=f"Improved convergence at the largest feasible n={n_best}",
    )

    print(
        f"  scaling best n={n_best}, m={best_model['m']}, agents={best_model['n_agents']}, "
        f"T_long={T_long}, best_dgd={best_dgd['final']:.3e}, best_gt={best_gt['final']:.3e}"
    )
    return n_best, records


def run():
    """Execute the full Part I experimental suite."""
    ensure_dir(str(FIGURES_DIR))
    first_db = DATA_DIR / "first_database.pkl"
    if not first_db.exists():
        raise FileNotFoundError("Missing data/first_database.pkl")

    x, y = load_first_database(first_db)
    problem = build_nystrom_problem(x, y, n=N_PART1, m=M_PART1, selection=True, seed=SEED)
    alpha_star = solve_centralized(problem["K_nm"], problem["y_n"], problem["K_mm"], sigma=SIGMA, nu=NU)
    obj_star = objective(alpha_star, problem["K_nm"], problem["y_n"], problem["K_mm"], sigma=SIGMA, nu=NU)
    L_central, mu_central = smoothness_and_strong_convexity(problem["K_nm"], problem["K_mm"], sigma=SIGMA, nu=NU)
    agents = make_agent_data(problem, N_AGENTS, sigma=SIGMA, nu=NU)

    local_sizes = [agent["n_local"] for agent in agents]
    if local_sizes != [N_PART1 // N_AGENTS] * N_AGENTS:
        raise ValueError("The 100 points must be split evenly across the 5 agents.")

    print("Part I")
    print(f"  n = {N_PART1}, m = {M_PART1}, agents = {N_AGENTS}")
    print(f"  centralized objective = {obj_star:.6f}")
    print(f"  centralized L = {L_central:.6f}")
    print(f"  centralized mu = {mu_central:.6f}")

    save_dataset_plot(
        problem["x_n"],
        problem["y_n"],
        FIGURES_DIR / "part1_dataset.pdf",
        landmark_indices=problem["landmark_indices"],
    )

    line_histories, adj_line, W_line, _incidence_line, line_steps = run_line_baselines(problem, agents, alpha_star)
    graph_histories = run_graph_sweep(agents, alpha_star)
    run_break_and_push_sum(agents, alpha_star, adj_line, W_line, line_steps)
    n_best, scaling_records = run_scaling_suite(x, y)

    print(f"  line graph tau_dual = {line_steps['tau_dual']:.3e}")
    print(f"  line graph rho_admm = {line_steps['rho_admm']:.3e}")
    for label, hist in line_histories.items():
        print(f"  final {label} bar gap = {hist['bar_gap'][-1]:.6e}")

    generated = [
        "part1_dataset.pdf",
        "part1_gap_line.pdf",
        "part1_reconstruction_compare.pdf",
        "part1_dgd_graph_compare.pdf",
        "part1_gt_graph_compare.pdf",
        "part1_dual_graph_compare.pdf",
        "part1_admm_graph_compare.pdf",
        "part1_break_convergence.pdf",
        "part1_push_sum_recovery.pdf",
        "part1_scaling_final_gap.pdf",
        "part1_scaling_iterations.pdf",
        "part1_scaling_surrogates.pdf",
        "part1_scaling_functions.pdf",
        "part1_scaling_best_tuned.pdf",
    ]
    _ = graph_histories
    write_part1_summary(
        FIGURES_DIR / "part1_theory_summary.txt",
        generated,
        line_steps,
        scaling_records,
        n_best,
    )
