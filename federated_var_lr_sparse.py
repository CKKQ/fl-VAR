

"""Federated low-rank + sparse VAR estimator.

This module implements a clean GitHub-ready version of the two-stage federated
learning procedure for multi-client high-dimensional VAR(p) models.

Overview
--------
Stage I (shared representation learning with optional DP noise)
    - Each client computes a local gradient with respect to the shared low-rank
      parameter B0 = A0^T.
    - Optional Gaussian noise is added for privacy.
    - The noisy gradient is projected onto the tangent space of the rank-r
      manifold at the current iterate.
    - The server aggregates the projected gradients and updates B0, followed by
      a rank-r SVD projection.

Stage II (client-specific personalization)
    - Each client estimates a sparse deviation D_k = Delta_k^T by FISTA:

          min_D (1/T_k) ||Y_k - X_k (B0 + D)||_F^2 + varpi_k ||D||_1.

    - The final client-specific coefficient is A_k = A0 + Delta_k.

Conventions
-----------
We optimize in B-space, where B = A^T has shape (dp, d). For client k,
Y_k[t] = y_{k,t+p} and X_k[t] stacks the p lagged observations.

Example usage
-------------
>>> from var_dgp import DGPConfig, generate_dgp
>>> from federated_var_lr_sparse import StageIOptions, StageIIOptions, fit_federated_var
>>> cfg = DGPConfig(
...     d=5,
...     p=2,
...     client_T={1: 220, 2: 220, 3: 220},
...     r=2,
...     q=0.2,
...     s_q=40.0,
...     seed=2026,
... )
>>> dgp = generate_dgp(cfg)
>>> stage1 = StageIOptions(N_g=50, r=2, rho=0.1, add_dp_noise=False)
>>> stage2 = StageIIOptions(N_l=300, varpi=0.02)
>>> res = fit_federated_var(y_list=dgp["y"], p=cfg.p, stage1=stage1, stage2=stage2)
>>> len(res.A_hat_list)
3

This module depends only on numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from var_utils import build_var_regression_mats, soft_threshold, truncate_by_effective_T

Array = np.ndarray

@dataclass
class FederatedFitResult:
    """Container for the fitted federated VAR quantities."""
    A0_hat: Array
    Delta_hat_list: List[Array]
    A_hat_list: List[Array]
    B0_hat: Array
    stage1_history: List[Dict[str, float]]
    stage1_sigmas: List[float]
    stage2_diagnostics: List[Dict[str, object]]
    client_sample_sizes: List[int]
    diagnostics: Dict[str, object]



def project_rank_r(M: Array, r: int) -> Array:
    """Best rank-r approximation by truncated SVD."""
    r = int(r)
    if r <= 0:
        return np.zeros_like(M)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    rr = min(r, s.size)
    return (U[:, :rr] * s[:rr]) @ Vt[:rr, :]


def proj_tangent_rank_r(M: Array, G: Array, r: int) -> Array:
    """Project G onto the tangent space of the rank-r manifold at M."""
    r = int(r)
    if r <= 0:
        return np.zeros_like(G)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    rr = min(r, s.size)
    U = U[:, :rr]
    V = Vt.T[:, :rr]
    UU = U @ U.T
    VV = V @ V.T
    return UU @ G + G @ VV - UU @ G @ VV




def gaussian_mechanism_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Gaussian mechanism std using a standard sufficient bound."""
    epsilon = float(epsilon)
    delta = float(delta)
    sensitivity = float(sensitivity)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be nonnegative")
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


# =========================
# Stage I: shared representation learning
# =========================

@dataclass
class StageIOptions:
    """Options for Stage I federated representation learning."""

    N_g: int = 50
    rho: float = 0.1
    r: int = 3
    weighted: bool = True
    seed: int = 777
    verbose: bool = False
    print_every: int = 25

    # Privacy-related options
    add_dp_noise: bool = False
    epsilon: float = 3.0
    delta: float = 1e-5
    xi_list: Optional[Sequence[float]] = None
    xi_const: float = 1.0


@dataclass
class StageIIOptions:
    """Options for Stage II client-specific sparse refinement."""

    N_l: int = 500
    eta: float = 0.1
    eta_mode: str = "auto"  # "auto" or "fixed"
    varpi: float = 0.05
    varpi_list: Optional[Sequence[float]] = None
    selection: str = "fixed"  # "fixed" | "bic" | "val"
    varpi_grid: Optional[Sequence[float]] = None
    nnz_tol: float = 1e-8
    val_frac: float = 0.2
    verbose: bool = False


def local_grad_B0(X: Array, Y: Array, B0: Array) -> Array:
    """Gradient of (1/T)||Y - X B0||_F^2 with respect to B0."""
    T = X.shape[0]
    R = X @ B0 - Y
    return (2.0 / float(T)) * (X.T @ R)


def stage1_representation_learning(
    X_list: List[Array],
    Y_list: List[Array],
    opts: StageIOptions,
    B0_init: Optional[Array] = None,
) -> Dict[str, object]:
    """Run Stage I representation learning with optional DP noise.

    Args:
        X_list: Client design matrices.
        Y_list: Client response matrices.
        opts: Stage I options.
        B0_init: Optional initialization in B-space of shape (dp, d).

    Returns:
        Dictionary with keys `B0_hat`, `history`, and `sigmas`.
    """
    K = len(X_list)
    if K == 0:
        raise ValueError("X_list and Y_list must be non-empty")

    dp = X_list[0].shape[1]
    d = Y_list[0].shape[1]
    for k in range(K):
        if X_list[k].ndim != 2 or Y_list[k].ndim != 2:
            raise ValueError("Each X_k and Y_k must be 2D")
        if X_list[k].shape[0] != Y_list[k].shape[0]:
            raise ValueError("X_k and Y_k must have the same number of rows")
        if X_list[k].shape[1] != dp or Y_list[k].shape[1] != d:
            raise ValueError("All clients must share the same (dp, d)")

    N_g = int(opts.N_g)
    if N_g <= 0:
        raise ValueError("N_g must be positive")
    if opts.xi_list is not None and len(opts.xi_list) != N_g:
        raise ValueError("xi_list must have length N_g")

    rng = np.random.default_rng(int(opts.seed))
    B0 = np.zeros((dp, d), dtype=float) if B0_init is None else np.asarray(B0_init, dtype=float).copy()
    if B0.shape != (dp, d):
        raise ValueError("B0_init must have shape (dp, d)")

    T_list = [int(X.shape[0]) for X in X_list]
    T_tot = float(np.sum(T_list))
    scales = [float(Tk) / T_tot for Tk in T_list] if opts.weighted else [1.0 / K] * K

    S = np.zeros((dp, dp), dtype=float)
    for X in X_list:
        S += X.T @ X
    H_op = float((2.0 / T_tot) * np.linalg.norm(S, ord=2))
    rho_cap = 1.0 / (H_op + 1e-12)
    rho_eff = float(min(float(opts.rho), rho_cap))

    history: List[Dict[str, float]] = []
    sigmas: List[float] = []

    for n in range(N_g):
        if opts.add_dp_noise:
            xi_n = float(opts.xi_list[n]) if opts.xi_list is not None else float(opts.xi_const)
            sigma_n = gaussian_mechanism_sigma(
                epsilon=float(opts.epsilon),
                delta=float(opts.delta),
                sensitivity=xi_n,
            )
        else:
            sigma_n = 0.0
        sigmas.append(float(sigma_n))

        Z_sum = np.zeros_like(B0)
        g_fro_list: List[float] = []

        for k in range(K):
            gk = local_grad_B0(X_list[k], Y_list[k], B0)
            g_fro_list.append(float(np.linalg.norm(gk, ord="fro")))
            if sigma_n > 0.0:
                noise = rng.normal(loc=0.0, scale=sigma_n, size=B0.shape)
                gk = gk + noise
            proj_gk = proj_tangent_rank_r(M=B0, G=gk, r=int(opts.r))
            Z_sum += scales[k] * proj_gk

        B0_new = project_rank_r(B0 - rho_eff * Z_sum, r=int(opts.r))

        loss_num = 0.0
        for k in range(K):
            Rk = Y_list[k] - X_list[k] @ B0_new
            loss_num += float(np.sum(Rk * Rk))
        pooled_loss = float(loss_num / T_tot)

        history.append(
            {
                "iter": float(n),
                "sigma": float(sigma_n),
                "rho_eff": float(rho_eff),
                "avg_grad_fro": float(np.mean(g_fro_list)),
                "loss": float(pooled_loss),
            }
        )

        if opts.verbose and (n % int(opts.print_every) == 0 or n == N_g - 1):
            print(f"[Stage I] iter={n:4d} loss={pooled_loss:.6g}", flush=True)

        B0 = B0_new

    return {"B0_hat": B0, "history": history, "sigmas": sigmas}


# =========================
# Stage II: client-specific personalization
# =========================

def stage2_personalized_fista(
    X: Array,
    Y: Array,
    B0_hat: Array,
    opts: StageIIOptions,
    varpi_k: Optional[float] = None,
) -> Dict[str, object]:
    """Solve the Stage II sparse refinement problem for one client."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    B0_hat = np.asarray(B0_hat, dtype=float)

    T = X.shape[0]
    dp, d = B0_hat.shape
    if X.shape != (T, dp) or Y.shape != (T, d):
        raise ValueError("Incompatible shapes among X, Y, and B0_hat")

    varpi = float(opts.varpi if varpi_k is None else varpi_k)

    D = np.zeros((dp, d), dtype=float)
    D_tilde = D.copy()
    q = 1.0

    XTX = X.T @ X
    XTY = X.T @ Y

    eta_eff = float(opts.eta)
    if str(opts.eta_mode).lower() == "auto":
        opnorm = float(np.linalg.norm(X, ord=2))
        L = (2.0 / float(T)) * (opnorm ** 2)
        eta_eff = 1.0 / (L + 1e-12)

    def grad(Din: Array) -> Array:
        return (2.0 / float(T)) * (XTX @ (B0_hat + Din) - XTY)

    diag: Dict[str, List[float]] = {"obj": [], "step": []}

    for n in range(int(opts.N_l)):
        G = grad(D_tilde)
        D_next = soft_threshold(D_tilde - eta_eff * G, tau=eta_eff * varpi)
        if not np.isfinite(D_next).all():
            break

        q_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * q * q))
        D_tilde = D_next + ((q - 1.0) / (q_next + 1e-12)) * (D_next - D)

        if (n % 25) == 0 or n == int(opts.N_l) - 1:
            R = Y - X @ (B0_hat + D_next)
            loss = float(np.sum(R * R) / float(T))
            obj = float(loss + varpi * np.sum(np.abs(D_next)))
            diag["obj"].append(obj)
            diag["step"].append(float(np.linalg.norm(D_next - D, ord="fro")))

        D = D_next
        q = q_next

    B_hat = B0_hat + D
    return {"D_hat": D, "B_hat": B_hat, "diagnostics": diag, "varpi": varpi, "eta_eff": float(eta_eff)}


def bic_score_stage2(Y: Array, X: Array, B0_hat: Array, D_hat: Array, nnz_tol: float) -> float:
    """BIC for Stage II with fixed B0_hat and estimated sparse deviation D_hat."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    B0_hat = np.asarray(B0_hat, dtype=float)
    D_hat = np.asarray(D_hat, dtype=float)
    T, d = Y.shape
    R = Y - X @ (B0_hat + D_hat)
    rss = float(np.sum(R * R))
    df = int(np.sum(np.abs(D_hat) > float(nnz_tol)))
    return float(T * np.log(rss / (T * d) + 1e-12) + df * np.log(T))


def val_sq_loss_stage2(Y: Array, X: Array, B0_hat: Array, D_hat: Array) -> float:
    """Validation squared loss only, without the L1 penalty term."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    B0_hat = np.asarray(B0_hat, dtype=float)
    D_hat = np.asarray(D_hat, dtype=float)
    T = int(Y.shape[0])
    if T <= 0:
        return float("inf")
    R = Y - X @ (B0_hat + D_hat)
    return float(np.sum(R * R) / float(T))


def select_varpi_by_bic(X: Array, Y: Array, B0_hat: Array, opts: StageIIOptions) -> float:
    """Select varpi by BIC over opts.varpi_grid for one client."""
    if opts.varpi_grid is None or len(opts.varpi_grid) == 0:
        return float(opts.varpi)

    best_varpi = float(opts.varpi_grid[0])
    best_bic = float("inf")
    for v in opts.varpi_grid:
        out = stage2_personalized_fista(X, Y, B0_hat=B0_hat, opts=opts, varpi_k=float(v))
        D_hat = np.asarray(out["D_hat"], dtype=float)
        if not np.isfinite(D_hat).all():
            continue
        bic = bic_score_stage2(Y, X, B0_hat=B0_hat, D_hat=D_hat, nnz_tol=float(opts.nnz_tol))
        if bic < best_bic:
            best_bic = bic
            best_varpi = float(v)
    return float(best_varpi)


def select_varpi_by_val(X: Array, Y: Array, B0_hat: Array, opts: StageIIOptions) -> float:
    """Select varpi by validation squared loss over opts.varpi_grid for one client."""
    if opts.varpi_grid is None or len(opts.varpi_grid) == 0:
        return float(opts.varpi)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = int(X.shape[0])
    vf = min(max(float(opts.val_frac), 0.0), 0.5)
    n_val = int(np.ceil(vf * n))
    n_val = max(5, n_val) if n >= 10 else max(1, n_val)
    n_tr = n - n_val
    if n_tr < 5 or n_val < 1:
        return float(opts.varpi)

    Xtr, Ytr = X[:n_tr, :], Y[:n_tr, :]
    Xva, Yva = X[n_tr:, :], Y[n_tr:, :]

    best_varpi = float(opts.varpi_grid[0])
    best_val = float("inf")
    for v in opts.varpi_grid:
        out = stage2_personalized_fista(Xtr, Ytr, B0_hat=B0_hat, opts=opts, varpi_k=float(v))
        D_hat = np.asarray(out["D_hat"], dtype=float)
        if not np.isfinite(D_hat).all():
            continue
        val = val_sq_loss_stage2(Yva, Xva, B0_hat=B0_hat, D_hat=D_hat)
        if val < best_val:
            best_val = val
            best_varpi = float(v)
    return float(best_varpi)


# =========================
# Public API
# =========================

def fit_federated_var(
    y_list: Sequence[Array],
    p: int,
    stage1: StageIOptions,
    stage2: StageIIOptions,
    T_eff_list: Optional[Sequence[Optional[int]]] = None,
    B0_init: Optional[Array] = None,
) -> FederatedFitResult:
    """Fit the two-stage federated low-rank + sparse VAR model.

    This is the main high-level routine intended for actual observed multi-client
    data. Given one multivariate time series per client, the function:

    1. optionally truncates each client series to an effective length;
    2. builds client-wise VAR regression matrices;
    3. runs Stage I to estimate the shared low-rank component;
    4. runs Stage II separately on each client to estimate sparse deviations;
    5. returns the final federated coefficient estimates and diagnostics.

    Args:
        y_list: List of client time series, each of shape (T_k, d).
        p: VAR lag order.
        stage1: Options for Stage I.
        stage2: Options for Stage II.
        T_eff_list: Optional effective sample sizes, one per client.
        B0_init: Optional initialization of the shared parameter in B-space.

    Returns:
        FederatedFitResult containing:
        - `A0_hat`: shared low-rank component in A-space;
        - `Delta_hat_list`: client-specific sparse deviations;
        - `A_hat_list`: final client-specific coefficient matrices.
    """
    if len(y_list) == 0:
        raise ValueError("y_list must be non-empty")
    if p <= 0:
        raise ValueError("p must be positive")

    if T_eff_list is not None and len(T_eff_list) != len(y_list):
        raise ValueError("T_eff_list must have the same length as y_list")

    y_list_used: List[Array] = []
    for k, y in enumerate(y_list):
        T_eff = None if T_eff_list is None else T_eff_list[k]
        y_use = truncate_by_effective_T(np.asarray(y, dtype=float), p=p, T_eff=T_eff)
        y_list_used.append(y_use)

    X_list: List[Array] = []
    Y_list: List[Array] = []
    for y in y_list_used:
        Xk, Yk = build_var_regression_mats(y, p=p)
        X_list.append(Xk)
        Y_list.append(Yk)

    out1 = stage1_representation_learning(X_list=X_list, Y_list=Y_list, opts=stage1, B0_init=B0_init)
    B0_hat = np.asarray(out1["B0_hat"], dtype=float)
    A0_hat = B0_hat.T

    Delta_hat_list: List[Array] = []
    A_hat_list: List[Array] = []
    stage2_diags: List[Dict[str, object]] = []

    for k in range(len(y_list_used)):
        if stage2.varpi_list is not None:
            varpi_k = float(stage2.varpi_list[k])
        else:
            selection = str(stage2.selection).lower()
            if selection == "bic" and stage2.varpi_grid is not None:
                varpi_k = select_varpi_by_bic(X_list[k], Y_list[k], B0_hat=B0_hat, opts=stage2)
            elif selection == "val" and stage2.varpi_grid is not None:
                varpi_k = select_varpi_by_val(X_list[k], Y_list[k], B0_hat=B0_hat, opts=stage2)
            else:
                varpi_k = float(stage2.varpi)

            if stage2.verbose:
                print(f"[Stage II] client {k+1}: selected varpi_k={varpi_k:.3g}", flush=True)

        out2 = stage2_personalized_fista(X_list[k], Y_list[k], B0_hat=B0_hat, opts=stage2, varpi_k=varpi_k)
        D_hat = np.asarray(out2["D_hat"], dtype=float)
        B_hat = np.asarray(out2["B_hat"], dtype=float)

        Delta_hat = D_hat.T
        A_hat = B_hat.T
        Delta_hat_list.append(Delta_hat)
        A_hat_list.append(A_hat)
        stage2_diags.append(dict(out2))

    client_sample_sizes = [int(X.shape[0]) for X in X_list]
    diagnostics = {
        "K": int(len(y_list_used)),
        "d": int(Y_list[0].shape[1]),
        "p": int(p),
        "dp": int(X_list[0].shape[1]),
        "stage1_weighted": bool(stage1.weighted),
        "stage1_add_dp_noise": bool(stage1.add_dp_noise),
        "stage2_selection": str(stage2.selection),
    }

    return FederatedFitResult(
        A0_hat=A0_hat,
        Delta_hat_list=Delta_hat_list,
        A_hat_list=A_hat_list,
        B0_hat=B0_hat,
        stage1_history=list(out1["history"]),
        stage1_sigmas=list(out1["sigmas"]),
        stage2_diagnostics=stage2_diags,
        client_sample_sizes=client_sample_sizes,
        diagnostics=diagnostics,
    )