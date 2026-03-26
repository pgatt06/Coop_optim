from .config import (
    DATA_DIR,
    FEDAVG_BATCH,
    FEDAVG_EPOCHS,
    FEDAVG_REQUIRED_ROUNDS,
    FEDAVG_SELECTED_CLIENTS,
    FEDAVG_SWEEP_ROUNDS,
    FIGURES_DIR,
    M_PART2,
    NU,
    SCAFFOLD_ROUNDS,
    SCAFFOLD_SELECTED_CLIENTS,
    SEED,
    SIGMA,
)
from .data_utils import ensure_dir, load_second_database
from .federated import build_federated_problem, run_fedavg, run_scaffold
from .plotting import save_semilogy_plot


def run():
    """Execute the full Part II federated-learning suite."""
    ensure_dir(str(FIGURES_DIR))
    second_db = DATA_DIR / "second_database.pkl"
    if not second_db.exists():
        raise FileNotFoundError("Missing data/second_database.pkl")

    X, Y = load_second_database(second_db)
    fed_problem = build_federated_problem(X, Y, m=M_PART2, sigma=SIGMA, nu=NU)
    clients = fed_problem["clients"]
    client_sizes = [client["n"] for client in clients]
    if client_sizes != [20] * len(clients):
        raise ValueError("This setup expects 5 clients with 20 samples each.")

    L_max = max(client["L"] for client in clients)
    lr_const = 0.25 / L_max

    print("Part II")
    print(f"  clients = {len(clients)}, local sizes = {client_sizes}")
    print(f"  centralized objective = {fed_problem['objective_star']:.6f}")
    print(f"  local L_max = {L_max:.6f}")
    print(f"  FedAvg step = {lr_const:.6e}")

    fedavg_curves = {}
    for E in FEDAVG_EPOCHS:
        lr = lr_const if E < 50 else 0.5 * lr_const
        rounds = FEDAVG_REQUIRED_ROUNDS[E]
        _, curve = run_fedavg(
            clients=clients,
            alpha_star=fed_problem["alpha_star"],
            objective_fn=fed_problem["objective"],
            K_mm=fed_problem["K_mm"],
            sigma=SIGMA,
            nu=NU,
            rounds=rounds,
            B=FEDAVG_BATCH,
            C=FEDAVG_SELECTED_CLIENTS,
            E=E,
            lr0=lr,
            diminishing=False,
            seed=SEED,
        )
        fedavg_curves[f"E={E}"] = curve
        print(f"  FedAvg E={E}: final objective error = {curve[-1]:.6e}")

    save_semilogy_plot(
        fedavg_curves,
        FIGURES_DIR / "part2_fedavg_required_E.pdf",
        ylabel="Objective error",
        title="FedAvg - B=20, C=5",
    )

    sweep_curves = {}
    sweep_settings = [
        ("B=20, C=5, E=5, constant step", dict(B=20, C=5, E=5, diminishing=False, lr0=lr_const)),
        ("B=10, C=5, E=5, constant step", dict(B=10, C=5, E=5, diminishing=False, lr0=lr_const)),
        ("B=5, C=3, E=5, constant step", dict(B=5, C=3, E=5, diminishing=False, lr0=lr_const)),
        ("B=20, C=5, E=50, decreasing step", dict(B=20, C=5, E=50, diminishing=True, lr0=lr_const)),
    ]
    for label, params in sweep_settings:
        _, curve = run_fedavg(
            clients=clients,
            alpha_star=fed_problem["alpha_star"],
            objective_fn=fed_problem["objective"],
            K_mm=fed_problem["K_mm"],
            sigma=SIGMA,
            nu=NU,
            rounds=FEDAVG_SWEEP_ROUNDS,
            seed=SEED + 1,
            **params,
        )
        sweep_curves[label] = curve

    save_semilogy_plot(
        sweep_curves,
        FIGURES_DIR / "part2_fedavg_param_sweep.pdf",
        ylabel="Objective error",
        title="FedAvg - effect of B, C, E, and the step schedule",
    )

    _, fedavg_curve = run_fedavg(
        clients=clients,
        alpha_star=fed_problem["alpha_star"],
        objective_fn=fed_problem["objective"],
        K_mm=fed_problem["K_mm"],
        sigma=SIGMA,
        nu=NU,
        rounds=SCAFFOLD_ROUNDS,
        B=FEDAVG_BATCH,
        C=SCAFFOLD_SELECTED_CLIENTS,
        E=5,
        lr0=lr_const,
        diminishing=False,
        seed=SEED + 2,
    )
    _, scaffold_curve = run_scaffold(
        clients=clients,
        alpha_star=fed_problem["alpha_star"],
        objective_fn=fed_problem["objective"],
        K_mm=fed_problem["K_mm"],
        sigma=SIGMA,
        nu=NU,
        rounds=SCAFFOLD_ROUNDS,
        B=FEDAVG_BATCH,
        C=SCAFFOLD_SELECTED_CLIENTS,
        E=5,
        lr=lr_const,
        seed=SEED + 2,
    )
    print(f"  optional FedAvg (C=3, E=5): final objective error = {fedavg_curve[-1]:.6e}")
    print(f"  optional SCAFFOLD: final objective error = {scaffold_curve[-1]:.6e}")
