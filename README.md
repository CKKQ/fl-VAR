# Private Federated Learning for High-dimensional Time Series

This repository accompanies the paper **"Private Federated Learning for High-dimensional Time Series"**.
It implements the proposed **federated low-rank + sparse VAR** framework together with supporting modules for simulation, benchmark estimation, and empirical evaluation.
The central methodological contribution of the repository is the federated estimator implemented in `federated_var_lr_sparse.py`. The remaining files provide supporting functionality for simulation design, benchmark comparison, and real-data analysis.

## Repository structure

### `federated_var_lr_sparse.py`

This file constitutes the **core methodological module** of the paper. It implements the proposed **two-stage federated estimator**, which is the main contribution of the repository.

The procedure is organized into two stages:

- **Stage I:** estimation of the shared low-rank structure across clients;
- **Stage II:** estimation of client-specific sparse refinements.

### `var_dgp.py`

This file implements the data-generating process used in the simulation studies. In particular, it can generate:

- a shared low-rank component `A0`;
- client-specific sparse deviations `Delta_k`;
- synthetic multivariate VAR($p$) time series.

### `var_utils.py`

This file contains shared utility functions used throughout the repository. Its purpose is to collect common operations that are repeatedly needed by both the single-client and federated procedures. Its main functionality includes:

- construction of VAR regression matrices;
- truncation by effective sample size;
- soft-thresholding.

### `single_client_var_lr_sparse.py`

This file implements the **single-client structured estimator**, which is solved via **ADMM**. 

### `benchmarks.py`

This file implements the benchmark estimators used for comparison. It collects the principal non-federated or simpler alternative procedures against which the proposed method is evaluated. The benchmark class includes:

- OLS;
- nuclear-norm penalized regression;
- $l_1$-penalized regression.

### `evaluation.py`

This file implements the empirical evaluation pipeline. Its main functionality includes:

- loading processed datasets;
- rolling or expanding forecasting evaluation;
- RMSFE-based comparison of the federated method and the benchmark estimators.

### `national/` and `state/`

These folders contain the processed empirical datasets used in the paper. In particular, the series have already undergone the preprocessing steps described in the paper (e.g., seasonal adjustment and the prescribed transformations).

- `national/`: national-level empirical dataset;
- `state/`: state-level empirical datasets.

The `state/` folder includes not only the five states analyzed in the paper, but also the remaining U.S. states. Consequently, the repository supports both replication of the reported empirical analysis and potential extensions to broader state-level applications.

### Typical workflows

In simulation studies, a typical workflow is as follows:

1. generate data using `var_dgp.py`;
2. fit the federated method using `federated_var_lr_sparse.py`;
3. compare it with the single-client estimator and the benchmark methods.

In empirical applications, a typical workflow is:

1. use the processed datasets in `national/` or `state/`;
2. conduct estimation and comparison through `evaluation.py`;
3. assess forecasting performance using RMSFE.
