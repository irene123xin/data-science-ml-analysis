# Dataset setup

Raw datasets are not committed to this repository. The California Housing data is loaded through scikit-learn; the retail workbook and MovieLens files should be downloaded from their original publishers.

## California Housing

- Source: 1990 US census data distributed through `sklearn.datasets.fetch_california_housing`.
- Documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>
- Local file: none. Scikit-learn downloads and caches the dataset on first use when it is not already available.
- Target: median house value in units of $100,000.

## UCI Online Retail

- Source: UCI Machine Learning Repository, **Online Retail** dataset.
- Dataset page: <https://archive.ics.uci.edu/dataset/352/online+retail>
- Direct workbook: <https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx>
- Expected local path: `data/online-retail.xlsx`

The notebook does not redistribute the 23 MB workbook. It removes rows without customer identifiers or descriptions, duplicate rows, cancelled invoices, and transactions with non-positive quantity or unit price before creating RFM features.

## MovieLens latest-small

- Source: GroupLens Research, MovieLens `ml-latest-small`.
- Download page: <https://grouplens.org/datasets/movielens/latest/>
- Archive: <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>
- License and README: included by GroupLens in the downloaded archive.
- Expected directory: `data/ml-latest-small/`
- Files used: `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`.

The source archive used for the original analysis contained these four CSV files, corresponding to 100,836 ratings, 9,742 movies, and 610 users. They are excluded from this public copy so redistribution remains governed by the publisher's terms.
