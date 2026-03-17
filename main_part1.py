import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from coop_optim.config import (
    SIGMA,
    NU,
    SEED,
    N_PART1,
    M_PART1,
    N_AGENTS,
    GRID_SIZE,
    MAX_ITERS,
    BREAK_ITERS,
    P_LOSS,
    P_ACTIVE,
    N_SCALING_MIN,
    N_SCALING_MAX,
    N_SCALING_THRESHOLD,
    N_SCALING_T_PROBE,
    N_SCALING_GROWTH,
    N_SCALING_MAX_EVALS,
    N_SCALING_BATCH_SIZE,
    N_SCALING_PER_EVAL_TIME_LIMIT,
    N_SCALING_TOTAL_TIME_BUDGET,
    N_SCALING_MAX_GROWTH,
    N_SCALING_PROGRESS_RATIO,
)
from coop_optim.data_utils import load_first_database, ensure_dir
from coop_optim.centralized import build_nystrom_problem, solve_centralized, objective, smoothness_and_strong_convexity
from coop_optim.distributed_objectives import make_agent_data
from coop_optim.graphs import get_graph
from coop_optim.algorithms import (
    run_dgd,
    run_gradient_tracking,
    run_dual_decomposition,
    run_consensus_admm,
    make_directed_row_stochastic,
    run_dgd_packet_loss,
    run_dual_decomposition_packet_loss,
    run_async_dgd,
    build_column_stochastic_pushsum,
    run_push_sum_dgd_directed,
)
from coop_optim.plotting import save_history_plot, save_prediction_plot
from coop_optim.kernel_utils import predict_from_alpha
from coop_optim.scaling import find_largest_n_possible

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')


def _directed_matrix_for_part1(W_line, n_agents, seed):
    if n_agents == 5:
        return np.array(
            [
                [0.70, 0.30, 0.00, 0.00, 0.00],
                [0.00, 0.60, 0.40, 0.00, 0.00],
                [0.00, 0.00, 0.80, 0.20, 0.00],
                [0.00, 0.00, 0.00, 0.55, 0.45],
                [0.50, 0.00, 0.00, 0.00, 0.50],
            ],
            dtype=float,
        )
    return make_directed_row_stochastic(W_line, seed=seed)


def run():
    ensure_dir(FIG_DIR)
    first_db = os.path.join(DATA_DIR, 'first_database.pkl')
    if not os.path.exists(first_db):
        raise FileNotFoundError('Put first_database.pkl inside data/.')

    x, y = load_first_database(first_db)
    problem = build_nystrom_problem(x, y, n=N_PART1, m=M_PART1, selection=True, seed=SEED)
    alpha_star = solve_centralized(problem['M'], problem['y_n'], problem['Kmm'], sigma=SIGMA, nu=NU)
    obj_star = objective(alpha_star, problem['M'], problem['y_n'], problem['Kmm'], sigma=SIGMA, nu=NU)
    L, mu = smoothness_and_strong_convexity(problem['M'], problem['Kmm'], sigma=SIGMA, nu=NU)
    agents = make_agent_data(problem, N_AGENTS)

    print('=== Part I ===')
    print(f'n = {N_PART1}, m = {M_PART1}, agents = {N_AGENTS}')
    print(f'Centralized objective = {obj_star:.6f}')
    print(f'Strong convexity mu = {mu:.6f}')
    print(f'Smoothness L = {L:.6f}')

    graph_names = ['cycle', 'line', 'complete']
    x_grid = np.linspace(-1.0, 1.0, GRID_SIZE)

    for graph_name in graph_names:
        _, W = get_graph(graph_name, N_AGENTS)

        step_dgd = 0.001
        step_gt = 0.001
        step_dual = 0.01
        rho_admm = 1.0

        hist_dgd = run_dgd(agents, W, alpha_star, step=step_dgd, n_iters=MAX_ITERS, seed=SEED, random_init=False)
        hist_gt = run_gradient_tracking(agents, W, alpha_star, step=step_gt, n_iters=MAX_ITERS, seed=SEED, random_init=False)
        hist_dd = run_dual_decomposition(agents, W, alpha_star, step=step_dual, n_iters=MAX_ITERS)
        hist_admm = run_consensus_admm(agents, W, alpha_star, rho=rho_admm, n_iters=MAX_ITERS)

        save_history_plot(
            {
                'DGD': hist_dgd['max_gap'],
                'Gradient tracking': hist_gt['max_gap'],
                'Dual decomposition': hist_dd['max_gap'],
                'ADMM': hist_admm['max_gap'],
            },
            os.path.join(FIG_DIR, f'part1_gap_{graph_name}.pdf'),
            ylabel=r'Max $\|\alpha_i^t-\alpha^*\|$',
            title=f'Part I - {graph_name} graph',
        )

        save_prediction_plot(
            problem['x_n'],
            problem['y_n'],
            x_grid,
            {
                'Centralized': (problem['x_m'], alpha_star),
                'DGD': (problem['x_m'], hist_dgd['alphas'][0]),
                'GT': (problem['x_m'], hist_gt['alphas'][0]),
                'Dual decomposition': (problem['x_m'], hist_dd['alphas'][0]),
                'ADMM': (problem['x_m'], hist_admm['alphas'][0]),
            },
            os.path.join(FIG_DIR, f'part1_prediction_{graph_name}.pdf'),
            title=f'Function reconstruction - {graph_name} graph',
        )

        print(f'[{graph_name}] final max gaps')
        print(f'  DGD   : {hist_dgd["max_gap"][-1]:.6e}')
        print(f'  GT    : {hist_gt["max_gap"][-1]:.6e}')
        print(f'  DD    : {hist_dd["max_gap"][-1]:.6e}')
        print(f'  ADMM  : {hist_admm["max_gap"][-1]:.6e}')

    # =========================
    # Rupture de convergence
    # =========================
    _, W_line = get_graph('line', N_AGENTS)
    W_directed = _directed_matrix_for_part1(W_line, N_AGENTS, SEED)
    step_break = 0.001

    hist_baseline = run_dgd(
        agents,
        W_line,
        alpha_star,
        step=step_break,
        n_iters=BREAK_ITERS,
        seed=SEED,
        random_init=False,
    )
    hist_directed = run_dgd(
        agents,
        W_directed,
        alpha_star,
        step=step_break,
        n_iters=BREAK_ITERS,
        seed=SEED,
        random_init=False,
    )
    hist_loss = run_dgd_packet_loss(
        agents,
        W_line,
        alpha_star,
        step=step_break,
        n_iters=BREAK_ITERS,
        p_loss=P_LOSS,
        seed=SEED,
        random_init=False,
    )
    
    
    hist_async = run_async_dgd(
        agents,
        W_line,
        alpha_star,
        step=step_break,
        n_iters=BREAK_ITERS,
        p_active=P_ACTIVE,
        seed=SEED,
        random_init=False,
    )

    save_history_plot(
        {
            'DGD non orienté': hist_baseline['mean_gap'],
            'DGD orienté': hist_directed['mean_gap'],
            'DGD pertes de paquets': hist_loss['mean_gap'],
            'DGD asynchrone': hist_async['mean_gap'],
        },
        os.path.join(FIG_DIR, 'part1_break_convergence.pdf'),
        ylabel=r'Mean $\|\alpha_i^t-\alpha^*\|$',
        title='Part I - rupture de convergence',
    )

    
    
    
    # ===========================
    # Dual decomposition - pertes de paquets
    # ===========================
    hist_dd_baseline = run_dual_decomposition(
        agents, 
        W_line, 
        alpha_star, 
        step=step_dual, 
        n_iters=BREAK_ITERS
    )
    hist_dd_directed = run_dual_decomposition(
        agents, 
        W_directed, 
        alpha_star, 
        step=step_dual, 
        n_iters=BREAK_ITERS
    )
    # Version Duale avec perte de paquets
    hist_loss_dual = run_dual_decomposition_packet_loss(
        agents,
        W_line,
        alpha_star,
        step=step_dual, # Utilise le step_dual défini plus haut (0.01)
        n_iters=BREAK_ITERS,
        p_loss=P_LOSS,
        seed=SEED
    )
    
    save_history_plot(
        {
            'DC non orienté': hist_dd_baseline['mean_gap'],
            'DC orienté': hist_dd_directed['mean_gap'],
            'DC pertes de paquets': hist_loss_dual['mean_gap'],
        },
        os.path.join(FIG_DIR, 'part1_break_convergence_dc.pdf'),
        ylabel=r'Mean $\|\alpha_i^t-\alpha^*\|$',
        title='Part I - rupture de convergence',
    )
    
    # =========================
    # Push-sum (cas orienté)
    # =========================
    Lmax_local = max(float(np.linalg.eigvalsh(ag['Q'])[-1]) for ag in agents)
    step_ps = 0.45 / Lmax_local
    P_col = build_column_stochastic_pushsum(W_directed)
    hist_ps = run_push_sum_dgd_directed(
        agents,
        P_col,
        alpha_star,
        step=step_ps,
        n_iters=BREAK_ITERS,
        seed=SEED,
        random_init=False,
    )

    save_history_plot(
        {
            # 'DGD orienté': hist_directed['mean_gap'],
            # 'Push-sum DGD': hist_ps['mean_gap'],
            'DGD orienté': hist_directed['max_gap'],
            'Push-sum DGD': hist_ps['max_gap'],
        },
        os.path.join(FIG_DIR, 'part1_pushsum_recovery.pdf'),
        ylabel=r'Max $\|\alpha_i^t-\alpha^*\|$',
        title='Part I - récupération push-sum',
    )

    # =========================
    # Montée en n (m = ceil(sqrt(n)))
    # =========================
    n_max_try = len(x) if N_SCALING_MAX <= 0 else min(int(N_SCALING_MAX), len(x))
    try:
        n_best, scaling_logs, model_cache = find_largest_n_possible(
            x,
            y,
            n_min=int(N_SCALING_MIN),
            n_max=int(n_max_try),
            n_agents=int(N_AGENTS),
            sigma=float(SIGMA),
            nu=float(NU),
            threshold=float(N_SCALING_THRESHOLD),
            T_probe=int(N_SCALING_T_PROBE),
            growth=float(N_SCALING_GROWTH),
            max_evals=int(N_SCALING_MAX_EVALS),
            batch_size=int(N_SCALING_BATCH_SIZE),
            per_eval_time_limit=float(N_SCALING_PER_EVAL_TIME_LIMIT),
            total_time_budget=float(N_SCALING_TOTAL_TIME_BUDGET),
            max_growth=float(N_SCALING_MAX_GROWTH),
            progress_ratio=float(N_SCALING_PROGRESS_RATIO),
            seed=int(SEED),
        )
        print(f'Largest feasible n found: {n_best}')
    except RuntimeError as exc:
        print(f'N-scaling skipped: {exc}')
        n_best = None
        scaling_logs = []
        model_cache = {}

    logs_csv = os.path.join(FIG_DIR, 'part1_n_scaling_logs.csv')
    fieldnames = sorted({k for row in scaling_logs for k in row.keys()}) if scaling_logs else []
    if fieldnames:
        with open(logs_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in scaling_logs:
                writer.writerow(row)

    records = sorted([row for row in scaling_logs if bool(row.get('feasible', False))], key=lambda r: int(r['n']))
    if records:
        plt.figure()
        plt.loglog([r['n'] for r in records], [r['dgd_final'] for r in records], marker='o', label='DGD final')
        plt.loglog([r['n'] for r in records], [r['gt_final'] for r in records], marker='o', label='GT final')
        plt.xlabel('n')
        plt.ylabel('final mean gap')
        plt.title('Part I - qualité de convergence vs n')
        plt.grid(True, which='both', ls=':', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'part1_n_scaling_final_gap.pdf'))
        plt.close()

        plt.figure()
        dgd_it = [np.nan if int(r['dgd_it']) < 0 else int(r['dgd_it']) for r in records]
        gt_it = [np.nan if int(r['gt_it']) < 0 else int(r['gt_it']) for r in records]
        plt.plot([r['n'] for r in records], dgd_it, marker='o', label='DGD iters to threshold')
        plt.plot([r['n'] for r in records], gt_it, marker='o', label='GT iters to threshold')
        plt.xlabel('n')
        plt.ylabel(f'iters to mean gap <= {N_SCALING_THRESHOLD}')
        plt.title('Part I - vitesse de convergence vs n')
        plt.grid(True, ls=':', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'part1_n_scaling_iterations.pdf'))
        plt.close()

        # Fonctions reconstruites pour quelques n faisables
        feasible_ns = [int(r['n']) for r in records]
        if len(feasible_ns) <= 6:
            n_plot = feasible_ns
        else:
            idx = np.linspace(0, len(feasible_ns) - 1, 6, dtype=int)
            n_plot = sorted(set(feasible_ns[i] for i in idx))

        plt.figure(figsize=(9.0, 5.0))
        for n_cur in n_plot:
            if n_cur not in model_cache:
                continue
            model = model_cache[n_cur]
            y_hat = predict_from_alpha(x_grid, model['x_sel'], model['alpha_star'])
            plt.plot(x_grid, y_hat, label=f'n={n_cur}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Part I - fonctions reconstruites (n croissant)')
        plt.grid(True, ls=':', alpha=0.4)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'part1_n_scaling_functions.pdf'))
        plt.close()


if __name__ == '__main__':
    run()
