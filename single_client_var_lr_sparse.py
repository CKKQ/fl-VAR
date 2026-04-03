"""Single-client learning for VAR(p) with low-rank + sparse decomposition.

We estimate (A0_k, Delta_k) for one client k by solving

    min_{A0, Delta}  (1/T) * sum_t || y_t - (A0 + Delta) x_t ||_2^2
                      + lambda * ||A0||_* + omega * ||Delta||_1
    s.t. ||A0||_inf <= zeta.

Implementation notes
--------------------
- We stack the regression in matrix form: Y = X B + E, where B = A^T and
  A in R^{d x (dp)}.
- We solve with ADMM on the constraint B = B0 + D.
- Tuning parameters can be selected by BIC, validation, or supplied directly.

Example usage
-------------
>>> from var_dgp import DGPConfig, generate_dgp
>>> from single_client_var_lr_sparse import suggest_penalty_grids, fit_single_client_var_lr_sparse
>>> cfg = DGPConfig(
...     d=5,
...     p=2,
...     client_T={1: 220},
...     r=2,
...     q=0.2,
...     s_q=40.0,
...     seed=2026,
... )
>>> dgp = generate_dgp(cfg)
>>> y = dgp["y"][0]
>>> lambda_grid, omega_grid, zeta = suggest_penalty_grids(y=y, p=cfg.p)
>>> res = fit_single_client_var_lr_sparse(
...     y=y,
...     p=cfg.p,
...     lambda_grid=lambda_grid,
...     omega_grid=omega_grid,
...     zeta=zeta,
...     selection="bic",
... )
>>> res.A_hat.shape
(5, 10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from var_utils import build_var_regression_mats, soft_threshold, truncate_by_effective_T


Array = np.ndarray


@dataclass
class FitResult:
    """Container for the fitted low-rank + sparse VAR coefficients."""
    A_hat: Array
    A0_hat: Array
    Delta_hat: Array
    lam: float
    omega: float
    bic: float
    nnz_hat: int
    rank_ratio_hat: int
    c_rank: float
    diagnostics: Dict[str, object]








def svt(Z: Array, tau: float) -> Tuple[Array, Array]:
    """Singular value thresholding: prox_{tau ||.||_*}(Z)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array")
    if not np.isfinite(Z).all():
        bad = np.logical_not(np.isfinite(Z))
        n_bad = int(np.sum(bad))
        raise ValueError(
            f"svt received non-finite entries (count={n_bad}); check ADMM stability / inputs"
        )

    tau = float(tau)
    if tau < 0:
        raise ValueError("tau must be nonnegative")

    m, n = Z.shape
    k = min(m, n)

    fro = float(np.linalg.norm(Z, ord="fro"))
    scale = max(1.0, fro)
    Zs = Z / scale

    try:
        U, s, Vt = np.linalg.svd(Zs, full_matrices=False)
        s = s * scale
        s_thr = np.maximum(s - tau, 0.0)
        return (U * (s_thr / scale)) @ Vt * scale, s
    except np.linalg.LinAlgError:
        if m <= n:
            G = Zs @ Zs.T
            w, U = np.linalg.eigh(G)
            idx = np.argsort(w)[::-1]
            w = np.clip(w[idx], 0.0, None)
            U = U[:, idx]
            s0 = np.sqrt(w)[:k]
            U = U[:, :k]
            Vt = np.zeros((k, n), dtype=float)
            tol = 1e-12
            for i in range(k):
                if s0[i] > tol:
                    Vt[i, :] = (U[:, i].T @ Zs) / s0[i]
            s = s0 * scale
        else:
            G = Zs.T @ Zs
            w, V = np.linalg.eigh(G)
            idx = np.argsort(w)[::-1]
            w = np.clip(w[idx], 0.0, None)
            V = V[:, idx]
            s0 = np.sqrt(w)[:k]
            V = V[:, :k]
            Vt = V.T
            tol = 1e-12
            U = np.zeros((m, k), dtype=float)
            for i in range(k):
                if s0[i] > tol:
                    U[:, i] = (Zs @ V[:, i]) / s0[i]
            s = s0 * scale

        s_thr = np.maximum(s - tau, 0.0)
        if k == 0:
            return np.zeros_like(Z), np.array([], dtype=float)
        Z_thr = (U * (s_thr / scale)) @ Vt * scale
        return Z_thr, s


def proj_inf_ball(A: Array, zeta: Optional[float]) -> Array:
    """Project onto {A: ||A||_inf <= zeta} by entrywise clipping."""
    if zeta is None:
        return A
    zeta = float(zeta)
    if zeta <= 0:
        raise ValueError("zeta must be positive")
    return np.clip(A, -zeta, zeta)


# =========================
# ADMM solver
# =========================

@dataclass
class ADMMOptions:
    rho: float = 1.0
    max_iter: int = 3000
    abstol: float = 1e-4
    reltol: float = 1e-3
    nnz_tol: float = 1e-8
    verbose: bool = False


def _admm_fit(
    Y: Array,
    X: Array,
    lam: float,
    omega: float,
    zeta: Optional[float],
    opts: ADMMOptions,
) -> Dict[str, Array]:
    """Fit one client for fixed (lam, omega) using ADMM."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.ndim != 2 or X.ndim != 2:
        raise ValueError("Y and X must be 2D arrays")
    T, d = Y.shape
    if X.shape[0] != T:
        raise ValueError("X and Y must have the same number of rows")
    dp = X.shape[1]
    if lam < 0 or omega < 0:
        raise ValueError("lam and omega must be nonnegative")

    rho = float(opts.rho)
    XtX = X.T @ X
    XtY = X.T @ Y
    H = (2.0 / T) * XtX + rho * np.eye(dp)

    try:
        L = np.linalg.cholesky(H)
        use_chol = True
    except np.linalg.LinAlgError:
        use_chol = False

    def solve_H(RHS: Array) -> Array:
        if use_chol:
            Z = np.linalg.solve(L, RHS)
            return np.linalg.solve(L.T, Z)
        return np.linalg.solve(H, RHS)

    B = np.zeros((dp, d), dtype=float)
    B0 = np.zeros_like(B)
    Dm = np.zeros_like(B)
    U = np.zeros_like(B)

    def fro_norm(A: Array) -> float:
        return float(np.linalg.norm(A, ord="fro"))

    for it in range(int(opts.max_iter)):
        V = B0 + Dm - U
        RHS = (2.0 / T) * XtY + rho * V
        B_new = solve_H(RHS)

        Z = B_new - Dm + U
        B0_tmp, _ = svt(Z, tau=lam / rho)
        B0_new = proj_inf_ball(B0_tmp, zeta=zeta)

        Z2 = B_new - B0_new + U
        D_new = soft_threshold(Z2, tau=omega / rho)

        R = B_new - B0_new - D_new
        U_new = U + R

        r_norm = fro_norm(R)
        s_norm = fro_norm(rho * (B0_new - B0))

        eps_pri = np.sqrt(B.size) * opts.abstol + opts.reltol * max(
            fro_norm(B_new), fro_norm(B0_new) + fro_norm(D_new)
        )
        eps_dual = np.sqrt(B.size) * opts.abstol + opts.reltol * fro_norm(rho * U_new)

        if opts.verbose and (it % 200 == 0 or it == opts.max_iter - 1):
            print(
                f"[ADMM] it={it:4d} r={r_norm:.3e} s={s_norm:.3e} "
                f"eps_pri={eps_pri:.3e} eps_dual={eps_dual:.3e}"
            )

        B, B0, Dm, U = B_new, B0_new, D_new, U_new

        if (r_norm <= eps_pri) and (s_norm <= eps_dual):
            break

    return {
        "B": B,
        "B0": B0,
        "D": Dm,
        "U": U,
        "n_iter": np.array([it + 1], dtype=int),
    }


# =========================
# Model selection
# =========================

def _effective_df_from_rank(B0: Array, Dm: Array, r_hat: int, nnz_tol: float) -> int:
    """Degrees of freedom for BIC using a provided rank estimate."""
    dp, d = B0.shape
    r_hat = int(max(0, min(r_hat, min(dp, d))))
    df_lr = r_hat * (dp + d - r_hat)
    nnz = int(np.sum(np.abs(Dm) > nnz_tol))
    return int(df_lr + nnz)


def bic_score(Y: Array, X: Array, B: Array, df: int) -> float:
    """BIC for multivariate regression Y ~ X B."""
    T, d = Y.shape
    R = Y - X @ B
    rss = float(np.sum(R * R))
    return T * np.log(rss / (T * d) + 1e-12) + df * np.log(T)


def compute_c_rank(d: int, T_k: int, p: int, c_scale: float = 1e-2) -> float:
    """Compute c(d, T_k) = 10^{-2} * sqrt(p d / T_k) for rank selection."""
    d = int(d)
    T_k = int(T_k)
    p = int(p)
    c_scale = float(c_scale)

    if d <= 0:
        raise ValueError("d must be positive")
    if T_k <= 0:
        raise ValueError("T_k must be positive")
    if p <= 0:
        raise ValueError("p must be positive")
    if c_scale <= 0:
        raise ValueError("c_scale must be positive")

    return float(c_scale * np.sqrt((p * d) / T_k) + 1e-12)


def estimate_rank_by_ratio(A0_hat: Array, c_rank: float, r_bar: int) -> Tuple[int, Array]:
    """Estimate rank via ridge-stabilized singular-value ratios."""
    r_bar = int(r_bar)
    if r_bar < 2:
        raise ValueError("r_bar must be >= 2")
    c_rank = float(c_rank)
    if c_rank <= 0:
        raise ValueError("c_rank must be positive")

    s = np.linalg.svd(np.asarray(A0_hat, dtype=float), compute_uv=False)
    if s.size == 0:
        return 0, np.array([], dtype=float)

    m = min(int(s.size), r_bar)
    if m < 2:
        return 0, np.array([], dtype=float)

    s_use = s[:m]
    ratios = (s_use[1:] + c_rank) / (s_use[:-1] + c_rank)
    r_hat = int(np.argmin(ratios) + 1)
    return r_hat, ratios



def _train_val_split(Y: Array, X: Array, val_frac: float) -> Tuple[Array, Array, Array, Array]:
    """Time-ordered split: first part for training, last part for validation."""
    if not (0.0 < val_frac < 0.5):
        raise ValueError("val_frac should be in (0, 0.5) for time-series validation")
    T = Y.shape[0]
    n_val = max(1, int(round(T * val_frac)))
    n_tr = T - n_val
    return Y[:n_tr], X[:n_tr], Y[n_tr:], X[n_tr:]


def _val_loss(Yv: Array, Xv: Array, B: Array) -> float:
    """Validation loss: average squared Frobenius loss."""
    R = Yv - Xv @ B
    return float(np.sum(R * R) / max(1, Yv.shape[0]))


def _auto_zeta_from_ols(Y: Array, X: Array, inflate: float = 3.0, mode: str = "rms") -> float:
    """Heuristic zeta choice using OLS coefficient scale."""
    B_ols, *_ = np.linalg.lstsq(X, Y, rcond=None)
    mode = str(mode).lower()
    if mode == "max":
        base = float(np.max(np.abs(B_ols)))
    elif mode == "rms":
        base = float(np.linalg.norm(B_ols, ord="fro") / np.sqrt(B_ols.size))
    else:
        raise ValueError("mode must be 'rms' or 'max'")
    return float(inflate * base + 1e-12)


def _auto_zeta_grid_from_ols(
    Y: Array,
    X: Array,
    mults: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    n_keep: int = 5,
    inflate: float = 3.0,
    mode: str = "rms",
) -> List[float]:
    """Construct a small zeta grid around the OLS-based zeta."""
    z0 = float(_auto_zeta_from_ols(Y, X, inflate=float(inflate), mode=str(mode)))
    cand = []
    for m in list(mults):
        try:
            mm = float(m)
        except Exception:
            continue
        if np.isfinite(mm) and mm > 0:
            z = z0 * mm
            if np.isfinite(z) and z > 0:
                cand.append(float(z))
    if len(cand) == 0:
        cand = [max(1e-12, z0)]
    cand = sorted(set(cand))
    if int(n_keep) > 0 and len(cand) > int(n_keep):
        cand = sorted(cand, key=lambda z: abs(z - z0))[: int(n_keep)]
        cand = sorted(set(cand))
    return cand


def suggest_penalty_grids(
    y: Array,
    p: int,
    n_lambda: int = 8,
    n_omega: int = 8,
    lam_span: Tuple[float, float] = (1e-2, 1e1),
    omg_span: Tuple[float, float] = (1e-3, 1e0),
    T_eff: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Suggest (lambda_grid, omega_grid, zeta_auto) from OLS scales."""
    y = truncate_by_effective_T(y, p=p, T_eff=T_eff)
    X, Y = build_var_regression_mats(y, p=p)
    B_ols, *_ = np.linalg.lstsq(X, Y, rcond=None)
    rms = float(np.linalg.norm(B_ols, ord="fro") / np.sqrt(B_ols.size) + 1e-12)

    lam_lo, lam_hi = lam_span
    omg_lo, omg_hi = omg_span
    lambda_grid = rms * np.logspace(np.log10(lam_lo), np.log10(lam_hi), n_lambda)
    omega_grid = rms * np.logspace(np.log10(omg_lo), np.log10(omg_hi), n_omega)

    zeta_auto = _auto_zeta_from_ols(Y, X, inflate=3.0, mode="rms")
    return lambda_grid, omega_grid, float(zeta_auto)


def fit_single_client_var_lr_sparse(
    y: Array,
    p: int,
    lambda_grid: Sequence[float],
    omega_grid: Sequence[float],
    zeta: Optional[float],
    zeta_grid: Optional[Sequence[float]] = None,
    zeta_mults: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    n_zeta: int = 5,
    zeta_inflate: float = 3.0,
    zeta_mode: str = "rms",
    opts: Optional[ADMMOptions] = None,
    selection: str = "bic",
    val_frac: float = 0.2,
    T_eff: Optional[int] = None,
    r_bar: int = 6,
    q: float = 0.2,
    s_q: float = 1.0,
    C_RSC: float = 1.0,
    c_scale: float = 1.0,
    return_path: bool = False,
) -> FitResult | Tuple[FitResult, List[Dict[str, object]]]:
    """Estimate (A0_k, Delta_k) for one client and select tuning parameters.

    This is the main high-level routine intended for actual observed data.
    Given one multivariate time series `y`, the function:

    1. optionally truncates the sample to an effective length `T_eff`;
    2. builds the VAR(p) regression matrices (Y, X);
    3. builds the regression inputs directly from the provided series;
    4. constructs or accepts candidate tuning values for zeta;
    5. searches over (lambda, omega, zeta) candidates;
    6. solves the low-rank + sparse decomposition for each candidate by ADMM;
    7. selects the final model by BIC, validation loss, or a fixed choice.

    Therefore, when you apply this file to a real dataset, this is typically the
    function you should call directly. The returned `FitResult` contains the final
    coefficient estimates:

    - `A_hat`: overall VAR coefficient matrix;
    - `A0_hat`: low-rank component;
    - `Delta_hat`: sparse component.

    If `return_path=True`, the function also returns the full tuning path for all
    tried parameter combinations.
    """
    if opts is None:
        opts = ADMMOptions()

    y = truncate_by_effective_T(y, p=p, T_eff=T_eff)
    X, Y = build_var_regression_mats(y, p=p)

    selection = str(selection).lower()
    if selection not in {"bic", "val", "fixed"}:
        raise ValueError("selection must be 'bic', 'val', or 'fixed'")

    if zeta is not None:
        zeta_cands: List[Optional[float]] = [float(zeta)]
    else:
        if zeta_grid is not None:
            zeta_cands = [float(v) for v in zeta_grid]
            zeta_cands = [v for v in zeta_cands if np.isfinite(v) and v > 0]
            if len(zeta_cands) == 0:
                raise ValueError("zeta_grid must contain at least one positive finite value")
        else:
            zeta_cands = _auto_zeta_grid_from_ols(
                Y,
                X,
                mults=zeta_mults,
                n_keep=int(n_zeta),
                inflate=float(zeta_inflate),
                mode=str(zeta_mode),
            )

    if selection == "val":
        Ytr, Xtr, Yv, Xv = _train_val_split(Y, X, val_frac=val_frac)
    else:
        Ytr, Xtr, Yv, Xv = Y, X, None, None

    lambda_grid = [float(v) for v in lambda_grid]
    omega_grid = [float(v) for v in omega_grid]
    if len(lambda_grid) == 0 or len(omega_grid) == 0:
        raise ValueError("lambda_grid and omega_grid must be non-empty")
    if any(v < 0 for v in lambda_grid) or any(v < 0 for v in omega_grid):
        raise ValueError("lambda_grid and omega_grid must be nonnegative")

    best: Optional[FitResult] = None
    path: List[Dict[str, object]] = []

    for zeta_cand in zeta_cands:
        zeta_use = None if zeta_cand is None else float(zeta_cand)
        for lam in lambda_grid:
            for omg in omega_grid:
                sol = _admm_fit(Y=Ytr, X=Xtr, lam=lam, omega=omg, zeta=zeta_use, opts=opts)
                B = sol["B"]
                B0 = sol["B0"]
                Dm = sol["D"]

                nnz_hat = int(np.sum(np.abs(Dm) > opts.nnz_tol))

                c_rank_cand = compute_c_rank(
                    d=Ytr.shape[1],
                    T_k=Ytr.shape[0],
                    p=p,
                    c_scale=c_scale,
                )
                r_ratio_cand, _ = estimate_rank_by_ratio(B0.T, c_rank=c_rank_cand, r_bar=r_bar)

                df = _effective_df_from_rank(B0=B0, Dm=Dm, r_hat=r_ratio_cand, nnz_tol=opts.nnz_tol)
                bic = bic_score(Y=Ytr, X=Xtr, B=B, df=df)

                vloss = None
                if selection == "val":
                    vloss = _val_loss(Yv, Xv, B)

                s = np.linalg.svd(B0, compute_uv=False)

                rec = {
                    "lambda": float(lam),
                    "omega": float(omg),
                    "zeta": None if zeta_use is None else float(zeta_use),
                    "bic": float(bic),
                    "val_loss": None if vloss is None else float(vloss),
                    "df": int(df),
                    "rank_ratio_hat": int(r_ratio_cand),
                    "c_rank": float(c_rank_cand),
                    "nnz_hat": int(nnz_hat),
                    "n_iter": int(sol["n_iter"][0]),
                    "nuc_norm_B0": float(np.sum(s)) if s.size else 0.0,
                    "l1_norm_D": float(np.sum(np.abs(Dm))),
                    "fro_B0": float(np.linalg.norm(B0, ord="fro")),
                    "fro_D": float(np.linalg.norm(Dm, ord="fro")),
                }
                path.append(rec)

                better = False
                if best is None:
                    better = True
                elif selection == "bic" and bic < best.bic:
                    better = True
                elif selection == "val" and vloss is not None and (
                    best.diagnostics["val_loss"] is None or vloss < best.diagnostics["val_loss"]
                ):
                    better = True
                elif selection == "fixed":
                    better = False

                if better:
                    A_hat = B.T
                    A0_hat = B0.T
                    Delta_hat = Dm.T
                    best = FitResult(
                        A_hat=A_hat,
                        A0_hat=A0_hat,
                        Delta_hat=Delta_hat,
                        lam=float(lam),
                        omega=float(omg),
                        bic=float(bic),
                        nnz_hat=int(nnz_hat),
                        rank_ratio_hat=int(r_ratio_cand),
                        c_rank=float(c_rank_cand),
                        diagnostics={
                            "selection": selection,
                            "val_frac": float(val_frac),
                            "val_loss": None if vloss is None else float(vloss),
                            "df": int(df),
                            "n_iter": int(sol["n_iter"][0]),
                            "dp": int(X.shape[1]),
                            "T_train": int(Ytr.shape[0]),
                            "T_total": int(Y.shape[0]),
                            "zeta": None if zeta_use is None else float(zeta_use),
                            "rank_ratio_hat": int(r_ratio_cand),
                            "c_rank": float(c_rank_cand),
                            "c_scale": float(c_scale),
                            "zeta_grid_used": [None if z is None else float(z) for z in zeta_cands],
                        },
                    )

                if selection == "fixed" and best is None:
                    A_hat = B.T
                    A0_hat = B0.T
                    Delta_hat = Dm.T
                    best = FitResult(
                        A_hat=A_hat,
                        A0_hat=A0_hat,
                        Delta_hat=Delta_hat,
                        lam=float(lam),
                        omega=float(omg),
                        bic=float(bic),
                        nnz_hat=int(nnz_hat),
                        rank_ratio_hat=int(r_ratio_cand),
                        c_rank=float(c_rank_cand),
                        diagnostics={
                            "selection": selection,
                            "val_frac": float(val_frac),
                            "val_loss": None,
                            "df": int(df),
                            "n_iter": int(sol["n_iter"][0]),
                            "dp": int(X.shape[1]),
                            "T_train": int(Ytr.shape[0]),
                            "T_total": int(Y.shape[0]),
                            "zeta": None if zeta_use is None else float(zeta_use),
                            "rank_ratio_hat": int(r_ratio_cand),
                            "c_rank": float(c_rank_cand),
                            "c_scale": float(c_scale),
                            "zeta_grid_used": [None if z is None else float(z) for z in zeta_cands],
                        },
                    )
                    if return_path:
                        return best, path
                    return best

    assert best is not None

    _, ratios = estimate_rank_by_ratio(best.A0_hat, c_rank=float(best.c_rank), r_bar=r_bar)
    best.diagnostics.update(
        {
            "r_bar": int(r_bar),
            "q": float(q),
            "s_q": float(s_q),
            "C_RSC": float(C_RSC),
            "c_scale": float(c_scale),
            "rank_ratios": ratios,
        }
    )

    if return_path:
        return best, path
    return best