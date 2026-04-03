"""Shared utilities for VAR-based estimators.

This module collects small helper functions that are used by both the
single-client and federated low-rank + sparse VAR estimators.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


def build_var_regression_mats(y: Array, p: int, center: bool = True) -> Tuple[Array, Array]:
    """Build VAR(p) regression matrices (X, Y) from one time series.

    Args:
        y: Array of shape (T, d).
        p: Lag order.
        center: Whether to center the series before building lags.

    Returns:
        X: Array of shape (T-p, dp).
        Y: Array of shape (T-p, d).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must be a 2D array of shape (T, d)")
    T, d = y.shape
    if p <= 0:
        raise ValueError("p must be positive")
    if T <= p:
        raise ValueError("Need T > p to build VAR regression matrices")

    if center:
        y = y - y.mean(axis=0, keepdims=True)

    n = T - p
    Y = y[p:, :].copy()
    X = np.zeros((n, d * p), dtype=float)
    for j in range(p):
        X[:, j * d : (j + 1) * d] = y[p - 1 - j : T - 1 - j, :]
    return X, Y


def truncate_by_effective_T(y: Array, p: int, T_eff: Optional[int] = None) -> Array:
    """Truncate a raw series so that VAR regression uses exactly T_eff rows."""
    if T_eff is None:
        return np.asarray(y, dtype=float)
    T_eff = int(T_eff)
    if T_eff <= 0:
        raise ValueError("T_eff must be a positive integer")
    y = np.asarray(y, dtype=float)
    need = T_eff + int(p)
    if y.shape[0] < need:
        raise ValueError(f"y has length {y.shape[0]} but need at least T_eff+p={need}")
    return y[:need, :]


def soft_threshold(M: Array, tau: float) -> Array:
    """Entrywise soft-thresholding."""
    tau = float(tau)
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)