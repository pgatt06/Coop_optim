from .algorithms import run_dgd, run_dgd_dp
from .centralized import build_nystrom_problem, solve_centralized
from .config import DATA_DIR, DP_CLIP_NORM, DP_DELTA, DP_EPSILONS, DP_ITERS, FIGURES_DIR, M_PART1, N_AGENTS, N_PART1, NU, SEED, SIGMA
from .data_utils import ensure_dir, load_first_database
from .distributed_objectives import make_agent_data
from .graphs import get_graph, spectral_beta
from .plotting import save_agent_gap_grid_plot


def run():
    ensure_dir(str(FIGURES_DIR))
    first_db = DATA_DIR / "first_database.pkl"
    if not first_db.exists():
        raise FileNotFoundError("Missing data/first_database.pkl")

    x, y = load_first_database(first_db)
    problem = build_nystrom_problem(x, y, n=N_PART1, m=M_PART1, selection=True, seed=SEED)
    alpha_star = solve_centralized(problem["K_nm"], problem["y_n"], problem["K_mm"], sigma=SIGMA, nu=NU)
    agents = make_agent_data(problem, N_AGENTS, sigma=SIGMA, nu=NU)

    _, W_line = get_graph("line", N_AGENTS, seed=SEED)
    L_max = max(agent["L"] for agent in agents)
    beta = spectral_beta(W_line)
    eta_dgd = min(0.9 * (1.0 - beta) / L_max, 0.9 / L_max)

    print("Part III")
    print(f"  line-graph beta = {beta:.6f}")
    print(f"  DGD step = {eta_dgd:.6e}")

    baseline = run_dgd(agents, W_line, alpha_star, step=eta_dgd, n_iters=DP_ITERS, seed=SEED)
    histories = {"Non-private DGD": baseline["agent_gaps"]}
    print(f"  non-private final bar gap = {baseline['bar_gap'][-1]:.6e}")

    for epsilon in DP_EPSILONS:
        hist = run_dgd_dp(
            agents,
            W_line,
            alpha_star,
            step=eta_dgd,
            epsilon=epsilon,
            n_iters=DP_ITERS,
            delta=DP_DELTA,
            clip_norm=DP_CLIP_NORM,
            seed=SEED,
        )
        histories[f"epsilon={epsilon}"] = hist["agent_gaps"]
        print(f"  epsilon={epsilon}: noise_std = {hist['noise_std']:.6e}, final bar gap = {hist['bar_gap'][-1]:.6e}")

    save_agent_gap_grid_plot(
        histories,
        FIGURES_DIR / "part3_dgd_dp_epsilons.pdf",
        ylabel=r"$\|\alpha_i^t-\alpha^\star\|$",
        title="Part III - private DGD",
    )
