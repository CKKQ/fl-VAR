"""Evaluation helpers for real-data federated and benchmark VAR comparisons.

This module extracts the reusable real-data comparison logic from the original
pipeline and focuses on RMSFE-based forecasting evaluation.

Included components
-------------------
- transformed real-data CSV loading helpers;
- optional global start/end date truncation;
- RMSFE-based forecast-error metrics;
- OLS baseline for single-client VAR regression;
- nuclear-norm and L1 penalized matrix-regression baselines via ADMM;
- simple time-ordered validation split;
- lambda-grid suggestion from data scale;
- rolling one-step-ahead evaluation for LS / nuclear / L1 baselines.

Conventions
-----------
All regressions are written in matrix form:

    Y = X B + E,

where X has shape (n, dp), Y has shape (n, d), and B has shape (dp, d).

This module is intended for real-data evaluation only. It does not include any
simulation or DGP utilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from benchmarks import BaselineOptions, fit_penalized_baseline, ols_fit
from federated_var_lr_sparse import StageIOptions, StageIIOptions, fit_federated_var
from var_utils import build_var_regression_mats

Array = np.ndarray


# =========================
# Real-data loading helpers
# =========================

def read_stationary_csv(path: str) -> pd.DataFrame:
    """Read a transformed CSV and standardize the time column to TIME_PERIOD.

    Supported time-column names:
    - TIME_PERIOD
    - Date
    - time_period
    """
    df = pd.read_csv(path)

    if "TIME_PERIOD" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "TIME_PERIOD"})
        elif "time_period" in df.columns:
            df = df.rename(columns={"time_period": "TIME_PERIOD"})
        else:
            raise ValueError(f"Missing 'TIME_PERIOD' (or 'Date') column in {path}")

    tp = df["TIME_PERIOD"].astype(str).str.strip()
    q_pat = r"^(\d{4})-?Q([1-4])$"
    if tp.notna().mean() > 0 and (tp.str.match(q_pat).mean() >= 0.95):
        tp2 = tp.str.replace(q_pat, r"\1Q\2", regex=True)
        df["TIME_PERIOD"] = pd.PeriodIndex(tp2, freq="Q").to_timestamp(how="start")
    else:
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce", format=None)

    df = df.dropna(subset=["TIME_PERIOD"]).sort_values("TIME_PERIOD")

    for c in df.columns:
        if c == "TIME_PERIOD":
            continue
        df[c] = df[c].astype(float)

    return df


def parse_start_date(s: str | None) -> pd.Timestamp | None:
    """Parse a global start date in YYYY-MM-DD format."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Bad start_date '{s}'. Use YYYY-MM-DD.")
    return pd.Timestamp(ts)


def parse_end_date(s: str | None) -> pd.Timestamp | None:
    """Parse a global end date in YYYY-MM-DD format."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Bad end_date '{s}'. Use YYYY-MM-DD.")
    return pd.Timestamp(ts)


def load_real_data_clients(
    *,
    states: Sequence[str],
    data_dir: str,
    p: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, object]:
    """Load transformed real-data CSVs and return aligned client arrays.

    This reproduces the reusable data-loading part of the real-data pipeline:
    - reads per-client transformed CSVs;
    - standardizes the time column;
    - optionally truncates to a global start/end date;
    - checks that feature columns match across clients.
    """
    states = [str(s).strip() for s in states if str(s).strip()]
    if len(states) == 0:
        raise ValueError("states must be non-empty")

    start_ts = parse_start_date(start_date)
    end_ts = parse_end_date(end_date)

    client_dfs: List[pd.DataFrame] = []
    feat_cols: List[str] | None = None

    for st in states:
        path = f"{data_dir}/{st}.csv"
        df = read_stationary_csv(path)

        if start_ts is not None:
            df = df[df["TIME_PERIOD"] >= start_ts].copy()
        if end_ts is not None:
            df = df[df["TIME_PERIOD"] <= end_ts].copy()

        if df.shape[0] <= int(p) + 10:
            raise ValueError(f"Too few rows for {st}: {df.shape[0]} (p={p})")

        cols = [c for c in df.columns if c != "TIME_PERIOD"]
        if feat_cols is None:
            feat_cols = cols
        elif cols != feat_cols:
            raise ValueError(f"Feature columns mismatch for {st}. Ensure all clients have identical columns.")

        client_dfs.append(df)

    assert feat_cols is not None
    y_list = [df[feat_cols].to_numpy(dtype=float) for df in client_dfs]
    n_obs = [int(y.shape[0]) for y in y_list]

    return {
        "states": states,
        "feature_columns": feat_cols,
        "dataframes": client_dfs,
        "y_list": y_list,
        "n_obs": n_obs,
    }


# =========================
# Error metrics
# =========================

def error_metrics(E: Array) -> Dict[str, float]:
    """Compute MSFE and RMSFE from an error matrix E.

    The averaging is over all elements (time x variables).
    """
    if E is None or np.asarray(E).size == 0:
        return {
            "msfe": float("nan"),
            "rmsfe": float("nan"),
        }

    E = np.asarray(E, dtype=float)
    msfe = float(np.mean(E**2))
    rmsfe = float(np.sqrt(msfe))
    return {"msfe": msfe, "rmsfe": rmsfe}




# =========================
# Rolling evaluation
# =========================

def evaluate_single_vs_baselines_from_regression(
    *,
    names: Sequence[str],
    X_list: Sequence[Array],
    Y_list: Sequence[Array],
    eval_last_n: int = 20,
    window: str = "expanding",
    window_size: Optional[int] = None,
    options: Optional[BaselineOptions] = None,
    seed: int = 777,
) -> Dict[str, object]:
    """Rolling 1-step-ahead evaluation for LS vs nuclear/L1 baselines."""
    if options is None:
        options = BaselineOptions()

    names = list(names)
    if len(names) == 0:
        raise ValueError("names must be non-empty")
    if len(X_list) != len(names) or len(Y_list) != len(names):
        raise ValueError("names, X_list, and Y_list must have the same length")

    d = int(np.asarray(Y_list[0]).shape[1])
    n_rows_list = [int(np.asarray(X).shape[0]) for X in X_list]

    eval_origins_map: Dict[str, List[int]] = {}
    window_size_map: Dict[str, int] = {}
    for nm, nr in zip(names, n_rows_list):
        n_eval_k = int(max(1, min(int(eval_last_n), nr - 1)))
        eval_start_k = int(max(1, nr - n_eval_k))
        eval_origins_map[nm] = list(range(eval_start_k, nr))
        if window_size is not None:
            window_size_map[nm] = int(window_size)
        else:
            window_size_map[nm] = int(max(20, nr - n_eval_k))

    n_steps = int(max(len(v) for v in eval_origins_map.values()))
    E_ls: Dict[str, List[Array]] = {nm: [] for nm in names}
    E_nuc: Dict[str, List[Array]] = {nm: [] for nm in names}
    E_l1: Dict[str, List[Array]] = {nm: [] for nm in names}

    for step in range(n_steps):
        active: List[int] = []
        t_by_k: Dict[int, int] = {}
        for k, nm in enumerate(names):
            origins = eval_origins_map[nm]
            if step < len(origins):
                active.append(k)
                t_by_k[k] = int(origins[step])

        if len(active) == 0:
            continue

        Xtr_list: List[Array] = []
        Ytr_list: List[Array] = []
        for k in active:
            nm = names[k]
            t_k = int(t_by_k[k])
            w = int(window_size_map[nm])
            ts = int(max(0, t_k - w)) if str(window).lower() == "rolling" else 0
            te = int(t_k)
            Xtr_list.append(np.asarray(X_list[k], dtype=float)[ts:te, :])
            Ytr_list.append(np.asarray(Y_list[k], dtype=float)[ts:te, :])

        for idx_in_pool, k in enumerate(active):
            nm = names[k]
            t_k = int(t_by_k[k])
            Xk = np.asarray(X_list[k], dtype=float)
            Yk = np.asarray(Y_list[k], dtype=float)
            y_true = Yk[t_k, :]

            Xtr = Xtr_list[idx_in_pool]
            Ytr = Ytr_list[idx_in_pool]

            B_ls = ols_fit(Xtr, Ytr)
            y_pred_ls = Xk[t_k, :] @ B_ls
            E_ls[nm].append((y_true - y_pred_ls).reshape(1, -1))

            fit_n = fit_penalized_baseline(
                Xtr,
                Ytr,
                kind="nuclear",
                options=options,
                warm_seed=int(seed) + 100000 * int(step) + 1000 * int(k) + 11,
            )
            B_n = np.asarray(fit_n["B_hat"], dtype=float)
            y_pred_n = Xk[t_k, :] @ B_n
            E_nuc[nm].append((y_true - y_pred_n).reshape(1, -1))

            fit_l = fit_penalized_baseline(
                Xtr,
                Ytr,
                kind="l1",
                options=options,
                warm_seed=int(seed) + 100000 * int(step) + 1000 * int(k) + 22,
            )
            B_l = np.asarray(fit_l["B_hat"], dtype=float)
            y_pred_l = Xk[t_k, :] @ B_l
            E_l1[nm].append((y_true - y_pred_l).reshape(1, -1))

    per_client: Dict[str, Dict[str, float]] = {}
    ls_stack: List[Array] = []
    nuc_stack: List[Array] = []
    l1_stack: List[Array] = []

    for nm in names:
        El = np.vstack(E_ls[nm]) if len(E_ls[nm]) else np.zeros((0, d))
        En = np.vstack(E_nuc[nm]) if len(E_nuc[nm]) else np.zeros((0, d))
        E1 = np.vstack(E_l1[nm]) if len(E_l1[nm]) else np.zeros((0, d))

        mls = error_metrics(El)
        mn = error_metrics(En)
        m1 = error_metrics(E1)

        per_client[nm] = {
            "ls_rmsfe": float(mls["rmsfe"]),
            "nuclear_rmsfe": float(mn["rmsfe"]),
            "l1_rmsfe": float(m1["rmsfe"]),
        }

        if El.size:
            ls_stack.append(El)
        if En.size:
            nuc_stack.append(En)
        if E1.size:
            l1_stack.append(E1)

    agg_ls = error_metrics(np.vstack(ls_stack) if ls_stack else np.zeros((0, d)))
    agg_n = error_metrics(np.vstack(nuc_stack) if nuc_stack else np.zeros((0, d)))
    agg_l1 = error_metrics(np.vstack(l1_stack) if l1_stack else np.zeros((0, d)))

    aggregate = {
        "ls_rmsfe": float(agg_ls["rmsfe"]),
        "nuclear_rmsfe": float(agg_n["rmsfe"]),
        "l1_rmsfe": float(agg_l1["rmsfe"]),
    }

    return {
        "per_client": per_client,
        "aggregate": aggregate,
        "n_rows_by_client": dict(zip(names, n_rows_list)),
    }


def evaluate_single_vs_baselines(
    *,
    names: Sequence[str],
    y_list: Sequence[Array],
    p: int,
    eval_last_n: int = 20,
    window: str = "expanding",
    window_size: Optional[int] = None,
    options: Optional[BaselineOptions] = None,
    seed: int = 777,
) -> Dict[str, object]:
    """Convenience wrapper that first builds VAR regression matrices from raw series."""
    X_list: List[Array] = []
    Y_list: List[Array] = []
    for y in y_list:
        Xk, Yk = build_var_regression_mats(np.asarray(y, dtype=float), p=int(p))
        X_list.append(Xk)
        Y_list.append(Yk)

    return evaluate_single_vs_baselines_from_regression(
        names=names,
        X_list=X_list,
        Y_list=Y_list,
        eval_last_n=int(eval_last_n),
        window=str(window),
        window_size=window_size,
        options=options,
        seed=int(seed),
    )




def _slice_raw_training_series(
    y: Array,
    *,
    p: int,
    t_reg: int,
    window: str,
    window_size: Optional[int],
) -> Array:
    """Slice a raw series so the resulting training sample matches the desired regression window.

    Here `t_reg` is the forecast origin in regression-row indexing, i.e. the row
    index in the matrices returned by `build_var_regression_mats`.
    """
    y = np.asarray(y, dtype=float)
    end_raw = int(t_reg + p)
    if str(window).lower() == "rolling" and window_size is not None:
        w = int(window_size)
        start_raw = max(0, end_raw - (w + p))
    else:
        start_raw = 0
    return y[start_raw:end_raw, :]



def evaluate_federated_vs_baselines(
    *,
    names: Sequence[str],
    y_list: Sequence[Array],
    p: int,
    stage1: StageIOptions,
    stage2: StageIIOptions,
    eval_last_n: int = 20,
    window: str = "expanding",
    window_size: Optional[int] = None,
    options: Optional[BaselineOptions] = None,
    seed: int = 777,
) -> Dict[str, object]:
    """Rolling 1-step-ahead evaluation for federated VAR vs LS / nuclear / L1 baselines.

    This function uses the federated estimator on the active clients at each
    forecast origin and compares its one-step-ahead forecasts against the
    single-client LS / nuclear / L1 baselines.
    """
    if options is None:
        options = BaselineOptions()

    names = list(names)
    if len(names) == 0:
        raise ValueError("names must be non-empty")
    if len(y_list) != len(names):
        raise ValueError("names and y_list must have the same length")

    X_list: List[Array] = []
    Y_list: List[Array] = []
    for y in y_list:
        Xk, Yk = build_var_regression_mats(np.asarray(y, dtype=float), p=int(p))
        X_list.append(Xk)
        Y_list.append(Yk)

    d = int(np.asarray(Y_list[0]).shape[1])
    n_rows_list = [int(np.asarray(X).shape[0]) for X in X_list]

    eval_origins_map: Dict[str, List[int]] = {}
    window_size_map: Dict[str, int] = {}
    for nm, nr in zip(names, n_rows_list):
        n_eval_k = int(max(1, min(int(eval_last_n), nr - 1)))
        eval_start_k = int(max(1, nr - n_eval_k))
        eval_origins_map[nm] = list(range(eval_start_k, nr))
        if window_size is not None:
            window_size_map[nm] = int(window_size)
        else:
            window_size_map[nm] = int(max(20, nr - n_eval_k))

    n_steps = int(max(len(v) for v in eval_origins_map.values()))
    E_fed: Dict[str, List[Array]] = {nm: [] for nm in names}
    E_ls: Dict[str, List[Array]] = {nm: [] for nm in names}
    E_nuc: Dict[str, List[Array]] = {nm: [] for nm in names}
    E_l1: Dict[str, List[Array]] = {nm: [] for nm in names}

    for step in range(n_steps):
        active: List[int] = []
        t_by_k: Dict[int, int] = {}
        for k, nm in enumerate(names):
            origins = eval_origins_map[nm]
            if step < len(origins):
                active.append(k)
                t_by_k[k] = int(origins[step])

        if len(active) == 0:
            continue

        train_y_list: List[Array] = []
        for k in active:
            nm = names[k]
            t_k = int(t_by_k[k])
            w = int(window_size_map[nm])
            y_train = _slice_raw_training_series(
                np.asarray(y_list[k], dtype=float),
                p=int(p),
                t_reg=t_k,
                window=str(window),
                window_size=w,
            )
            if y_train.shape[0] <= int(p):
                raise ValueError(f"Training sample too short for client {nm} at step {step}")
            train_y_list.append(y_train)

        fed_out = fit_federated_var(
            y_list=train_y_list,
            p=int(p),
            stage1=stage1,
            stage2=stage2,
        )

        for idx_in_pool, k in enumerate(active):
            nm = names[k]
            t_k = int(t_by_k[k])
            Xk = np.asarray(X_list[k], dtype=float)
            Yk = np.asarray(Y_list[k], dtype=float)
            y_true = Yk[t_k, :]

            nm_window = int(window_size_map[nm])
            if str(window).lower() == "rolling":
                ts = int(max(0, t_k - nm_window))
            else:
                ts = 0
            te = int(t_k)
            Xtr = Xk[ts:te, :]
            Ytr = Yk[ts:te, :]

            B_ls = ols_fit(Xtr, Ytr)
            y_pred_ls = Xk[t_k, :] @ B_ls
            E_ls[nm].append((y_true - y_pred_ls).reshape(1, -1))

            fit_n = fit_penalized_baseline(
                Xtr,
                Ytr,
                kind="nuclear",
                options=options,
                warm_seed=int(seed) + 100000 * int(step) + 1000 * int(k) + 11,
            )
            B_n = np.asarray(fit_n["B_hat"], dtype=float)
            y_pred_n = Xk[t_k, :] @ B_n
            E_nuc[nm].append((y_true - y_pred_n).reshape(1, -1))

            fit_l = fit_penalized_baseline(
                Xtr,
                Ytr,
                kind="l1",
                options=options,
                warm_seed=int(seed) + 100000 * int(step) + 1000 * int(k) + 22,
            )
            B_l = np.asarray(fit_l["B_hat"], dtype=float)
            y_pred_l = Xk[t_k, :] @ B_l
            E_l1[nm].append((y_true - y_pred_l).reshape(1, -1))

            A_fed = np.asarray(fed_out.A_hat_list[idx_in_pool], dtype=float)
            B_fed = A_fed.T
            y_pred_fed = Xk[t_k, :] @ B_fed
            E_fed[nm].append((y_true - y_pred_fed).reshape(1, -1))

    per_client: Dict[str, Dict[str, float]] = {}
    fed_stack: List[Array] = []
    ls_stack: List[Array] = []
    nuc_stack: List[Array] = []
    l1_stack: List[Array] = []

    for nm in names:
        Ef = np.vstack(E_fed[nm]) if len(E_fed[nm]) else np.zeros((0, d))
        El = np.vstack(E_ls[nm]) if len(E_ls[nm]) else np.zeros((0, d))
        En = np.vstack(E_nuc[nm]) if len(E_nuc[nm]) else np.zeros((0, d))
        E1 = np.vstack(E_l1[nm]) if len(E_l1[nm]) else np.zeros((0, d))

        mf = error_metrics(Ef)
        mls = error_metrics(El)
        mn = error_metrics(En)
        m1 = error_metrics(E1)

        per_client[nm] = {
            "fed_rmsfe": float(mf["rmsfe"]),
            "ls_rmsfe": float(mls["rmsfe"]),
            "nuclear_rmsfe": float(mn["rmsfe"]),
            "l1_rmsfe": float(m1["rmsfe"]),
        }

        if Ef.size:
            fed_stack.append(Ef)
        if El.size:
            ls_stack.append(El)
        if En.size:
            nuc_stack.append(En)
        if E1.size:
            l1_stack.append(E1)

    agg_f = error_metrics(np.vstack(fed_stack) if fed_stack else np.zeros((0, d)))
    agg_ls = error_metrics(np.vstack(ls_stack) if ls_stack else np.zeros((0, d)))
    agg_n = error_metrics(np.vstack(nuc_stack) if nuc_stack else np.zeros((0, d)))
    agg_l1 = error_metrics(np.vstack(l1_stack) if l1_stack else np.zeros((0, d)))

    aggregate = {
        "fed_rmsfe": float(agg_f["rmsfe"]),
        "ls_rmsfe": float(agg_ls["rmsfe"]),
        "nuclear_rmsfe": float(agg_n["rmsfe"]),
        "l1_rmsfe": float(agg_l1["rmsfe"]),
    }

    return {
        "per_client": per_client,
        "aggregate": aggregate,
        "n_rows_by_client": dict(zip(names, n_rows_list)),
    }


def evaluate_single_vs_baselines_on_real_data(
    *,
    states: Sequence[str],
    data_dir: str,
    p: int,
    start_date: str | None = None,
    end_date: str | None = None,
    eval_last_n: int = 20,
    window: str = "expanding",
    window_size: Optional[int] = None,
    options: Optional[BaselineOptions] = None,
    seed: int = 777,
) -> Dict[str, object]:
    """One-call real-data evaluation wrapper for LS / nuclear / L1 baselines."""
    loaded = load_real_data_clients(
        states=states,
        data_dir=data_dir,
        p=int(p),
        start_date=start_date,
        end_date=end_date,
    )

    result = evaluate_single_vs_baselines(
        names=loaded["states"],
        y_list=loaded["y_list"],
        p=int(p),
        eval_last_n=int(eval_last_n),
        window=str(window),
        window_size=window_size,
        options=options,
        seed=int(seed),
    )

    result.update(
        {
            "feature_columns": loaded["feature_columns"],
            "n_obs": loaded["n_obs"],
        }
    )
    return result



def evaluate_federated_vs_baselines_on_real_data(
    *,
    states: Sequence[str],
    data_dir: str,
    p: int,
    stage1: StageIOptions,
    stage2: StageIIOptions,
    start_date: str | None = None,
    end_date: str | None = None,
    eval_last_n: int = 20,
    window: str = "expanding",
    window_size: Optional[int] = None,
    options: Optional[BaselineOptions] = None,
    seed: int = 777,
) -> Dict[str, object]:
    """One-call real-data evaluation wrapper for federated VAR vs baseline methods using RMSFE-based comparison."""
    loaded = load_real_data_clients(
        states=states,
        data_dir=data_dir,
        p=int(p),
        start_date=start_date,
        end_date=end_date,
    )

    result = evaluate_federated_vs_baselines(
        names=loaded["states"],
        y_list=loaded["y_list"],
        p=int(p),
        stage1=stage1,
        stage2=stage2,
        eval_last_n=int(eval_last_n),
        window=str(window),
        window_size=window_size,
        options=options,
        seed=int(seed),
    )

    result.update(
        {
            "feature_columns": loaded["feature_columns"],
            "n_obs": loaded["n_obs"],
        }
    )
    return result