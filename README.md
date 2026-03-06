# Coop Optim

Projet de régression noyau coopérative (cours d’optimisation distribuée).

Ce dépôt contient :
- une implémentation modulaire des briques utiles (`coop_optim/`)
- un script principal pour la Partie I (`main_part1.py`)
- un script de recherche du plus grand `n` faisable (`tools/largest_n_possible.py`)
- des notebooks d’exploration dans `explo/`

## État actuel
- Partie I : implémentée et exécutable.

## Données attendues
Les fichiers suivants doivent être présents dans `data/` :
- `data/first_database.pkl`
- `data/second_database.pkl`

## Lancement
Depuis la racine du repo :

### Partie I
```bash
python3 main_part1.py
```

Sorties :
- courbes de convergence : `figures/part1_gap_*.pdf`
- reconstructions de fonction : `figures/part1_prediction_*.pdf`

### Lanceur global
```bash
python3 run_all.py
```

Remarque : actuellement `run_all.py` exécute surtout la Partie I.

### Recherche du plus grand `n` faisable
```bash
python3 tools/largest_n_possible.py
```

Options minimales disponibles :
```bash
python3 tools/largest_n_possible.py --data data/first_database.pkl --out-dir .
```

Sorties typiques :
- `largest_n_search_logs.csv`
- `n_scaling_final_gap.pdf`
- `n_scaling_iterations.pdf`
- `best_model.npz`

## Paramètres importants
### Partie I
Dans [`coop_optim/config.py`](coop_optim/config.py) :
- `N_PART1`, `M_PART1`, `N_AGENTS`
- `MAX_ITERS`
- `SIGMA`, `NU`

### Recherche de `n` max
Dans [`tools/largest_n_possible.py`](tools/largest_n_possible.py), section “Configuration par défaut” :
- bornes de recherche (`N_MIN`, `N_MAX`)
- budget temps (`PER_EVAL_TIME_LIMIT`, `TOTAL_TIME_BUDGET`)
- critères de faisabilité (`MAX_GROWTH`, `PROGRESS_RATIO`)

## Structure du dépôt
```text
coop_optim/
├── coop_optim/                 # cœur du code (algos, graphes, objectifs, plots)
├── data/                       # datasets .pkl
├── figures/                    # sorties PDF principales
├── tools/                      # scripts utilitaires (ex: plus grand n)
├── explo/                      # notebooks et essais
├── main_part1.py               # pipeline Partie I
└── run_all.py                  # point d’entrée global (Partie I active)
```

## Notes pratiques
- Les figures sont sauvegardées en PDF.
- Le script `largest_n_possible.py` force un backend Matplotlib non interactif pour mieux tourner en terminal/sandbox.
- Si un run est trop long, réduire `MAX_ITERS` (Partie I) ou les budgets de `tools/largest_n_possible.py`.
