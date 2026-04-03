"""Benchmark estimators for VAR comparisons.

This module collects the benchmark methods themselves, rather than the rolling
real-data evaluation workflow. It is intended to provide reusable baseline
estimators that can be imported by `evaluation.py` or other scripts.

Included benchmark methods
--------------------------
- OLS baseline for single-client VAR regression;
- nuclear-norm penalized matrix regression via ADMM;
- L1-penalized matrix regression via ADMM;
- simple time-ordered validation split;
- lambda-grid suggestion from data scale;
- deterministic warm-start helpers.

Conventions
-----------
All regressions are written in matrix form:

    Y = X B + E,

where X has shape (n, dp), Y has shape (n, d), and B has shape (dp, d).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class BaselineOptions:
    """Options for penalized single-client benchmark fitting."""

    selection: str = "val"  # "val" or "insample"
    val_frac: float = 0.1
    n_lambda: int = 6
    admm_max_iter: int = 300
    admm_tol: float = 1e-6
    admm_rho: float = 1.0

    # Deterministic random warm-starts
    random_warmstart: bool = False
    warmstart_scale: float = 1e-3
    warmstart_jitter: float = 0.0


# =========================
# Basic benchmark fits
# =========================

def ols_fit(X: Array, Y: Array) -> Array:
    """Stable LS fit using `lstsq`. Returns B_hat with shape (dp, d)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.size == 0 or Y.size == 0:
        return np.zeros((X.shape[1], Y.shape[1]), dtype=float)
    B_hat, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return np.asarray(B_hat, dtype=float)


# =========================
# Prox operators
# =========================

def soft_threshold(Z: Array, tau: float) -> Array:
    """Entrywise soft-thresholding."""
    tau = float(max(0.0, tau))
    return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0.0)


def svt(Z: Array, tau: float) -> Array:
    """Singular-value thresholding."""
    tau = float(max(0.0, tau))
    if tau == 0.0:
        return np.asarray(Z, dtype=float)
    U, s, Vt = np.linalg.svd(np.asarray(Z, dtype=float), full_matrices=False)
    s2 = np.maximum(s - tau, 0.0)
    if np.all(s2 == 0):
        return np.zeros_like(Z, dtype=float)
    return (U * s2) @ Vt


# =========================
# Tuning helpers
# =========================

def time_split_train_val(
    X: Array,
    Y: Array,
    val_frac: float,
    min_val_rows: int = 12,
) -> Tuple[Array, Array, Optional[Array], Optional[Array]]:
    """Time-ordered split: last block is validation.

    If the series is too short, returns `(X, Y, None, None)`.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = int(X.shape[0])
    if n <= 2 * int(min_val_rows):
        return X, Y, None, None
    n_val = int(max(int(min_val_rows), round(float(val_frac) * n)))
    n_val = int(min(n_val, n - int(min_val_rows)))
    n_fit = int(n - n_val)
    if n_fit < int(min_val_rows) or n_val < int(min_val_rows):
        return X, Y, None, None
    return X[:n_fit, :], Y[:n_fit, :], X[n_fit:, :], Y[n_fit:, :]


def suggest_lambda_grid_from_data(
    X: Array,
    Y: Array,
    *,
    kind: str,
    n_lambda: int,
    span: Tuple[float, float] = (1e-3, 1.0),
) -> List[float]:
    """Suggest a geometric lambda grid based on the scale of X^T Y."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_lambda = int(max(2, n_lambda))
    XtY = X.T @ Y
    if kind == "l1":
        lam_max = float(np.max(np.abs(XtY)))
    elif kind == "nuclear":
        try:
            lam_max = float(np.linalg.norm(XtY, ord=2))
        except Exception:
            lam_max = float(np.linalg.svd(XtY, compute_uv=False)[0])
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    lam_max = max(lam_max, 1e-12)
    lo = max(float(span[0]), 1e-9)
    hi = max(float(span[1]), lo * 1.001)
    grid = np.geomspace(lam_max * hi, lam_max * lo, num=n_lambda)
    return sorted({float(x) for x in grid if np.isfinite(x) and x > 0.0})


def make_random_warmstart(p: int, d: int, *, seed: int, scale: float) -> Array:
    """Deterministic random warm-start for ADMM given a seed."""
    scale = float(scale)
    if scale <= 0:
        return np.zeros((p, d), dtype=float)
    rng = np.random.default_rng(int(seed))
    return (scale * rng.standard_normal(size=(p, d))).astype(float)


# =========================
# Penalized benchmark fits
# =========================

def admm_matrix_prox(
    X: Array,
    Y: Array,
    *,
    lam: float,
    prox_kind: str,
    rho: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    init_B: Optional[Array] = None,
    init_Z: Optional[Array] = None,
    init_U: Optional[Array] = None,
) -> Array:
    """ADMM for

        min_B 0.5 ||Y - X B||_F^2 + lam * P(B),

    where P is either elementwise L1 or nuclear norm.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    lam = float(lam)
    rho = float(rho)
    if lam < 0:
        raise ValueError(f"lam must be nonnegative, got {lam}")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")

    _, p = X.shape
    d = Y.shape[1]

    XtX = X.T @ X
    XtY = X.T @ Y
    A = XtX + rho * np.eye(p)
    try:
        L = np.linalg.cholesky(A)
        use_chol = True
    except Exception:
        use_chol = False

    def solve_A(RHS: Array) -> Array:
        if use_chol:
            tmp = np.linalg.solve(L, RHS)
            return np.linalg.solve(L.T, tmp)
        return np.linalg.solve(A, RHS)

    B = (
        np.asarray(init_B, dtype=float).copy()
        if init_B is not None and np.asarray(init_B).shape == (p, d)
        else np.zeros((p, d), dtype=float)
    )
    Z = (
        np.asarray(init_Z, dtype=float).copy()
        if init_Z is not None and np.asarray(init_Z).shape == (p, d)
        else np.zeros((p, d), dtype=float)
    )
    U = (
        np.asarray(init_U, dtype=float).copy()
        if init_U is not None and np.asarray(init_U).shape == (p, d)
        else np.zeros((p, d), dtype=float)
    )

    for _ in range(int(max_iter)):
        Z_prev = Z.copy()
        RHS = XtY + rho * (Z - U)
        B = solve_A(RHS)

        Q = B + U
        if prox_kind == "l1":
            Z = soft_threshold(Q, lam / rho)
        elif prox_kind == "nuclear":
            Z = svt(Q, lam / rho)
        else:
            raise ValueError(f"Unknown prox_kind '{prox_kind}'")

        U = U + (B - Z)

        r = B - Z
        s = rho * (Z - Z_prev)
        r_norm = float(np.linalg.norm(r, ord="fro"))
        s_norm = float(np.linalg.norm(s, ord="fro"))
        b_norm = float(np.linalg.norm(B, ord="fro"))
        z_norm = float(np.linalg.norm(Z, ord="fro"))
        u_norm = float(np.linalg.norm(U, ord="fro"))

        prim = r_norm / max(1.0, b_norm, z_norm)
        dual = s_norm / max(1.0, rho * u_norm)
        if prim <= float(tol) and dual <= float(tol):
            break

    return Z


def fit_penalized_baseline(
    Xtr: Array,
    Ytr: Array,
    *,
    kind: str,
    options: BaselineOptions,
    warm_seed: int = 777,
) -> Dict[str, object]:
    """Fit nuclear/L1 baseline with lambda chosen by validation or in-sample MSE."""
    kind = str(kind).lower()
    selection = str(options.selection).lower()

    X_fit, Y_fit, X_val, Y_val = time_split_train_val(
        Xtr, Ytr, val_frac=float(options.val_frac)
    )
    grid = suggest_lambda_grid_from_data(
        X_fit,
        Y_fit,
        kind=("nuclear" if kind == "nuclear" else "l1"),
        n_lambda=int(options.n_lambda),
        span=(1e-3, 1.0),
    )

    best_lam = float(grid[len(grid) // 2]) if len(grid) else 0.0
    best_score = float("inf")
    best_B: Optional[Array] = None
    prev_B: Optional[Array] = None

    for j, lam in enumerate(grid):
        init_B = None
        init_Z = None
        init_U = None

        if bool(options.random_warmstart):
            p_dim = int(X_fit.shape[1])
            d_dim = int(Y_fit.shape[1])
            if j == 0 or prev_B is None:
                init_B = make_random_warmstart(
                    p_dim,
                    d_dim,
                    seed=int(warm_seed),
                    scale=float(options.warmstart_scale),
                )
            else:
                init_B = np.asarray(prev_B, dtype=float).copy()
                if float(options.warmstart_jitter) > 0:
                    init_B = init_B + make_random_warmstart(
                        p_dim,
                        d_dim,
                        seed=int(warm_seed + j),
                        scale=float(options.warmstart_jitter),
                    )
            init_Z = init_B
            init_U = np.zeros_like(init_B)

        B_hat = admm_matrix_prox(
            X_fit,
            Y_fit,
            lam=float(lam),
            prox_kind=("nuclear" if kind == "nuclear" else "l1"),
            rho=float(options.admm_rho),
            max_iter=int(options.admm_max_iter),
            tol=float(options.admm_tol),
            init_B=init_B,
            init_Z=init_Z,
            init_U=init_U,
        )

        if bool(options.random_warmstart):
            prev_B = np.asarray(B_hat, dtype=float)

        if selection == "val" and X_val is not None and Y_val is not None:
            R = Y_val - X_val @ B_hat
        else:
            R = Y_fit - X_fit @ B_hat
        score = float(np.mean(R**2))

        if score < best_score:
            best_score = score
            best_lam = float(lam)
            best_B = B_hat

    if best_B is None:
        best_B = admm_matrix_prox(
            X_fit,
            Y_fit,
            lam=float(best_lam),
            prox_kind=("nuclear" if kind == "nuclear" else "l1"),
            rho=float(options.admm_rho),
            max_iter=int(options.admm_max_iter),
            tol=float(options.admm_tol),
        )

    return {
        "B_hat": np.asarray(best_B, dtype=float),
        "meta": {
            "kind": kind,
            "lambda": float(best_lam),
            "score": float(best_score),
            "grid": grid,
            "random_warmstart": bool(options.random_warmstart),
            "warmstart_scale": float(options.warmstart_scale),
            "warmstart_jitter": float(options.warmstart_jitter),
            "warm_seed": int(warm_seed),
        },
    }