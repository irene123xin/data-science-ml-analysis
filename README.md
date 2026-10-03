# Data Science & Machine Learning Analysis

Three applied studies covering supervised regression, customer segmentation, and recommendation systems.

The repository preserves the original exploratory work while strengthening model evaluation, documenting data decisions, and making local execution reproducible. It is an analytical portfolio repository rather than a production application.

## Projects

| Project | Problem | Methods | Evaluation |
|---|---|---|---|
| California Housing Regression | Predict median census-block house value | Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost | Cross-validation, R², MAE, RMSE, residuals |
| Customer Segmentation | Describe groups in observed retail behavior | Cleaning, RFM, log transform, scaling, K-Means, PCA | Inertia, silhouette, Davies–Bouldin, median profiles |
| Movie Recommendation | Rank unseen films for each user | TF-IDF content profiles, user-based collaborative filtering, weighted hybrid | Precision@10, Recall@10, NDCG@10, HitRate@10 |

## 1. California Housing Regression

[`01_housing_regression.ipynb`](notebooks/01_housing_regression.ipynb) explores geographic, demographic, and housing attributes before comparing a linear baseline with ensemble regressors. Preprocessing stays inside scikit-learn pipelines. Grid search and five-fold model comparison use only the training partition; the selected estimator is then evaluated once on the untouched test set.

The original complete run selected LightGBM and reported test R² **0.8463**, MAE **0.2952**, and RMSE **0.4488** in the dataset target scale. Residual and feature-importance analysis are retained. This result is an offline benchmark, not a current housing valuation model.

## 2. Customer Segmentation

[`02_customer_segmentation.ipynb`](notebooks/02_customer_segmentation.ipynb) cleans the UCI Online Retail transactions and creates customer-level RFM features:

- **Recency:** days since the latest valid purchase.
- **Frequency:** number of distinct valid invoices.
- **Monetary:** total retained transaction value.

The notebook log-transforms and standardizes RFM before K-Means. At `k=4`, silhouette is **0.3375** and Davies–Bouldin is **1.0086**. A two-cluster split scores better quantitatively (silhouette **0.4328**), while four clusters retain more useful behavioral distinctions. The notebook documents this interpretability trade-off rather than presenting four clusters as objectively correct.

| Cluster | Customers | Median recency | Median frequency | Median monetary value | Descriptive interpretation |
|---:|---:|---:|---:|---:|---|
| 0 | 713 | 8.0 | 10.0 | £3,722.82 | Recent, high-frequency, high observed value |
| 1 | 1,622 | 174.5 | 1.0 | £296.09 | Inactive or at-risk observed behavior |
| 2 | 837 | 17.0 | 2.0 | £480.31 | Recent, occasional observed behavior |
| 3 | 1,166 | 57.0 | 4.0 | £1,345.20 | Established, mid-value observed behavior |

These descriptions summarize the retained data; they are not predictive personas.

## 3. Movie Recommendation System

[`03_movie_recommendation.ipynb`](notebooks/03_movie_recommendation.ipynb) retains the original MovieLens exploration, genre/tag TF-IDF representation, cosine-similarity recommendations, user-based collaborative filtering, and 60/40 hybrid score. It adds a timestamp-aware leave-one-positive-out evaluation.

Users need at least 10 ratings and two ratings of 4.0 or higher. Each eligible user's latest positive interaction is held out, models are fit on the remaining interactions, and 10 unseen movies are ranked. Results were verified across 608 eligible users:

| Approach | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.0046 | 0.0461 | 0.0267 | 0.0461 |
| Content profile | 0.0008 | 0.0082 | 0.0034 | 0.0082 |
| User-based collaborative | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid | 0.0002 | 0.0016 | 0.0016 | 0.0016 |

The popularity baseline is strongest under this sparse protocol. The original hybrid remains useful as a construction example, but the held-out evidence does not support claiming that it improves ranking quality.

## Skills Demonstrated

- Exploratory data analysis and visualization
- Data cleaning and feature engineering
- Leakage-aware train/test design and cross-validation
- Regression metrics and residual analysis
- RFM analysis, scaling, clustering, PCA, and cluster diagnostics
- Content-based, collaborative, and hybrid recommendation logic
- Offline Top-K ranking evaluation and baseline comparison
- Reproducible notebook and dataset documentation

## Tech Stack

Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · seaborn · XGBoost · LightGBM · CatBoost · Jupyter · pytest

## Repository Structure

```text
notebooks/               Three analytical studies
src/recommender_eval.py  Reusable Top-K ranking metrics
tests/                   Lightweight metric tests
data/README.md           Dataset sources and local paths
docs/images/             Future portfolio figures
requirements.txt         Analysis environment
ENVIRONMENT.md           Reproducibility notes
```

## Reproducibility

Random seeds are fixed where the underlying estimator supports them. The housing notebook separates test data before tuning, the segmentation notebook documents its snapshot and transformations, and the recommendation notebook removes held-out interactions before fitting. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for execution details.

## Datasets

California Housing is loaded through scikit-learn. UCI Online Retail and MovieLens latest-small are downloaded from their publishers and stored under `data/`; raw files are ignored by Git. Exact sources and expected paths are in [`data/README.md`](data/README.md).

## Limitations

- Housing results use an historical tabular benchmark and random holdout, not temporal or geographic validation.
- K-Means relies on distance and shape assumptions; RFM is descriptive and omits intent, returns, and unobserved behavior.
- MovieLens relevance is defined by a rating threshold, and one held-out item produces a high-variance offline estimate.
- The recommenders do not address cold start, online feedback, or satisfaction; offline metrics do not guarantee user value.

## Running Locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Download the datasets first using `data/README.md`. Run lightweight metric tests with:

```bash
pytest -q
```

## Author

Yi Xin Ding
