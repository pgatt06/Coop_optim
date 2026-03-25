from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

SIGMA = 0.5
NU = 1.0
SEED = 7

N_PART1 = 100
M_PART1 = 10
N_AGENTS = 5
PART1_DGD_ITERS = 30000
PART1_GT_ITERS = 100000
PART1_DD_ITERS = 300000
PART1_ADMM_ITERS = 30000

M_PART2 = 10
FEDAVG_REQUIRED_ROUNDS = {1: 25000, 5: 4000, 50: 1200}
FEDAVG_SWEEP_ROUNDS = 4000
FEDAVG_BATCH = 20
FEDAVG_SELECTED_CLIENTS = 5
FEDAVG_EPOCHS = (1, 5, 50)
SCAFFOLD_ROUNDS = 4000
SCAFFOLD_SELECTED_CLIENTS = 3

DP_ITERS = 10000
DP_DELTA = 1e-5
DP_CLIP_NORM = 5.0
DP_EPSILONS = (0.1, 1.0, 10.0)

GRAPH_SPECS = [
    ("cycle", "cycle"),
    ("line", "line"),
    ("small_world", "small-world"),
    ("complete", "complete"),
]
