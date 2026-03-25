from .algorithms import (
    dual_lipschitz_constant,
    run_consensus_admm,
    run_dgd,
    run_dual_decomposition,
    run_gradient_tracking,
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
    PART1_ADMM_ITERS,
    PART1_DD_ITERS,
    PART1_DGD_ITERS,
    PART1_GT_ITERS,
    SEED,
    SIGMA,
)
from .data_utils import ensure_dir, load_first_database
from .distributed_objectives import make_agent_data
from .graphs import get_graph, incidence_matrix, spectral_beta
from .plotting import save_agent_gap_grid_plot, save_dataset_plot, save_history_plot


def part1_steps(agents, W, incidence):
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


def run():
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

    adj_line, W_line = get_graph("line", N_AGENTS, seed=SEED)
    incidence_line, _ = incidence_matrix(adj_line)
    line_steps = part1_steps(agents, W_line, incidence_line)

    hist_line_dgd = run_dgd(agents, W_line, alpha_star, step=line_steps["eta_dgd"], n_iters=PART1_DGD_ITERS, seed=SEED)
    hist_line_gt = run_gradient_tracking(
        agents, W_line, alpha_star, step=line_steps["eta_gt"], n_iters=PART1_GT_ITERS, seed=SEED
    )
    hist_line_dd = run_dual_decomposition(
        agents, incidence_line, alpha_star, step=line_steps["tau_dual"], n_iters=PART1_DD_ITERS
    )
    hist_line_admm = run_consensus_admm(
        agents, adj_line, alpha_star, rho=line_steps["rho_admm"], n_iters=PART1_ADMM_ITERS
    )

    save_agent_gap_grid_plot(
        {
            "DGD": hist_line_dgd["agent_gaps"],
            "Gradient tracking": hist_line_gt["agent_gaps"],
            "Dual decomposition": hist_line_dd["agent_gaps"],
            "ADMM": hist_line_admm["agent_gaps"],
        },
        FIGURES_DIR / "part1_gap_line.pdf",
        ylabel=r"$\|\alpha_i^t-\alpha^\star\|$",
        title="Part I - line graph",
    )

    dgd_by_graph = {}
    gt_by_graph = {}
    for graph_name, graph_label in GRAPH_SPECS:
        adj, W = get_graph(graph_name, N_AGENTS, seed=SEED)
        incidence, _ = incidence_matrix(adj)
        params = part1_steps(agents, W, incidence)
        hist_dgd = run_dgd(agents, W, alpha_star, step=params["eta_dgd"], n_iters=PART1_DGD_ITERS, seed=SEED)
        hist_gt = run_gradient_tracking(agents, W, alpha_star, step=params["eta_gt"], n_iters=PART1_GT_ITERS, seed=SEED)
        dgd_by_graph[graph_label] = hist_dgd["bar_gap"]
        gt_by_graph[graph_label] = hist_gt["bar_gap"]
        print(f"  [{graph_label}] beta = {params['beta']:.6f}, eta_dgd = {params['eta_dgd']:.3e}, eta_gt = {params['eta_gt']:.3e}")

    print(f"  line graph tau_dual = {line_steps['tau_dual']:.3e}")
    print(f"  line graph rho_admm = {line_steps['rho_admm']:.3e}")
    print(f"  final DGD bar gap = {hist_line_dgd['bar_gap'][-1]:.6e}")
    print(f"  final GT bar gap = {hist_line_gt['bar_gap'][-1]:.6e}")
    print(f"  final dual bar gap = {hist_line_dd['bar_gap'][-1]:.6e}")
    print(f"  final ADMM bar gap = {hist_line_admm['bar_gap'][-1]:.6e}")

    save_history_plot(
        dgd_by_graph,
        FIGURES_DIR / "part1_dgd_graph_compare.pdf",
        ylabel=r"$\|\bar{\alpha}^t-\alpha^\star\|$",
        title="DGD - graph effect on the averaged iterate",
    )
    save_history_plot(
        gt_by_graph,
        FIGURES_DIR / "part1_gt_graph_compare.pdf",
        ylabel=r"$\|\bar{\alpha}^t-\alpha^\star\|$",
        title="Gradient tracking - graph effect on the averaged iterate",
    )
