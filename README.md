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

This design reflects the central modeling principle of the paper: common structure is learned collaboratively across clients, while client-level heterogeneity is captured through sparse personalized adjustments.

### `var_dgp.py`

This file implements the data-generating process used in the simulation studies. More specifically, it is designed to generate synthetic multivariate VAR($p$) systems that are aligned with the structural assumptions studied in the paper.

In particular, it can generate:

- a shared low-rank component `A0`;
- client-specific sparse deviations `Delta_k`;
- synthetic multivariate VAR($p$) time series.

Accordingly, this module provides the simulation foundation for assessing estimation accuracy, forecasting performance, and the behavior of the proposed federated procedure under controlled settings.

### `var_utils.py`

This file contains shared utility functions used throughout the repository. Its purpose is to collect common low-level operations that are repeatedly needed by both the single-client and federated procedures.

Its main functionality includes:

- construction of VAR regression matrices;
- truncation by effective sample size;
- soft-thresholding.

By isolating these operations in a separate module, the repository avoids unnecessary duplication and maintains consistency across different estimation and evaluation routines.

### `single_client_var_lr_sparse.py`

This file implements the **single-client structured estimator**, which is solved via **ADMM**. For an individual client, it estimates a low-rank component, a sparse component, and the resulting full coefficient matrix under the low-rank + sparse VAR formulation.

From a methodological perspective, this module corresponds to the single-client analogue of the structured estimation problem considered in the paper. In the empirical and simulation analyses, it primarily serves as a single-client benchmark and as a natural comparison point for the federated method.

### `benchmarks.py`

This file implements the benchmark estimators used for comparison. It collects the principal non-federated or simpler alternative procedures against which the proposed method is evaluated. The benchmark class includes:

- OLS;
- nuclear-norm penalized regression;
- $l_1$-penalized regression.

As such, this module provides the baseline estimation methods required for systematic performance comparison in both simulation and empirical studies.

### `evaluation.py`

This file implements the empirical evaluation pipeline. Its main functionality includes:

- loading processed datasets;
- rolling or expanding forecasting evaluation;
- RMSFE-based comparison of the federated method and the benchmark estimators.

This module therefore serves as the bridge between estimation and empirical analysis, enabling the forecasting comparisons reported in the paper.

### `national/` and `state/`

These folders contain the processed empirical datasets used in the paper and therefore provide the data counterpart to the estimation and evaluation code.

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

These workflows illustrate the modular design of the repository: simulation, estimation, benchmarking, and empirical evaluation are separated conceptually, while remaining closely connected in implementation.
