import os
import numpy as np

from coop_optim.config import SIGMA, NU, SEED, N_PART1, M_PART1, N_AGENTS, GRID_SIZE, MAX_ITERS
from coop_optim.data_utils import load_first_database, ensure_dir
from coop_optim.centralized import build_nystrom_problem, solve_centralized, objective, smoothness_and_strong_convexity
from coop_optim.distributed_objectives import make_agent_data
from coop_optim.graphs import get_graph
from coop_optim.algorithms import run_dgd, run_gradient_tracking, run_dual_decomposition, run_consensus_admm
from coop_optim.plotting import save_history_plot, save_prediction_plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')


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


if __name__ == '__main__':
    run()
