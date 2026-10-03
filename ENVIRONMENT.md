# Reproducible environment

Python 3.11 is recommended. The notebooks use deterministic random seeds for train/test splitting, K-Means, tree ensembles, boosting models, and ranking holdouts where applicable.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the retail and MovieLens data as described in `data/README.md`, then start Jupyter from the repository root:

```bash
jupyter lab
```

## Optional model libraries

The housing notebook compares XGBoost, LightGBM, and CatBoost because they are part of the original analysis. They are listed in `requirements.txt`, but binary-wheel availability can vary by Python version and platform. If one cannot be installed, the linear and Random Forest portions remain runnable; do not compare the reduced run directly with the complete results table.

## Re-execution order

Each notebook is independent and should be run from top to bottom. The MovieLens ranking evaluation imports `src.recommender_eval`, so Jupyter should be launched from the repository root. The notebooks also handle being opened with `notebooks/` as the working directory.

## Reproducibility notes

- California Housing is downloaded and cached by scikit-learn on first use.
- The housing test partition is separated before tuning; cross-validation occurs only on training data.
- The segmentation reference date is one day after the latest retained transaction.
- MovieLens evaluation holds out each eligible user's latest rating of at least 4.0 and excludes that row from fitting.
- Exact timings and some floating-point results can vary slightly across platforms and package versions.
