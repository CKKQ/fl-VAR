

"""Multi-client DGP for federated low-rank + sparse VAR(p) models.

This module generates synthetic data for a multi-client VAR(p) system with the
shared-plus-personalized structure

    A_k = A0 + Delta_k,

where:
- A0 is a common low-rank coefficient matrix;
- Delta_k is a client-specific weakly sparse deviation.

Model
-----
At client k,

    y_{k,t} = A_{k,1} y_{k,t-1} + ... + A_{k,p} y_{k,t-p} + eps_{k,t},

with

    A_k = [A_{k,1}, ..., A_{k,p}] in R^{d x (dp)}.

Key features
------------
- K clients with potentially different sample sizes T_k;
- common lag order p;
- exactly low-rank shared component A0;
- weakly sparse client-specific deviations Delta_k;
- optional client-specific innovation covariance matrices;
- optional stability enforcement through companion-matrix rescaling.

Example usage
-------------
>>> cfg = DGPConfig(
...     d=10,
...     p=2,
...     client_T={1: 300, 2: 350, 3: 320},
...     r=2,
...     q=0.2,
...     s_q=80.0,
...     seed=2026,
... )
>>> out = generate_dgp(cfg)
>>> len(out["y"])
3
>>> out["A0"].shape
(10, 20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


@dataclass
class DGPConfig:
    """Configuration for the multi-client low-rank + sparse VAR(p) DGP."""

    # Core dimensions
    d: int
    p: int

    # Client sample sizes
    # Preferred usage: provide client_T = {client_id: T_k}
    # Backward-compatible usage: provide (K, T_list)
    client_T: Optional[Dict[int, int]] = None
    K: Optional[int] = None
    T_list: Optional[Sequence[int]] = None

    # Derived / cached fields
    client_ids: List[int] = field(init=False)

    # Simulation controls
    burn_in: int = 200

    # Structure of A_k = A0 + Delta_k
    r: int = 3
    q: float = 0.0
    s_q: float = 200.0

    # Innovation covariance
    # None: identity for all clients
    # "diag": random diagonal covariance for each client
    # sequence: explicit list of (d,d) covariance matrices
    Sigma_eps: Optional[Union[str, Sequence[Array]]] = None

    # Stability control
    enforce_stability: bool = True
    target_rho: float = 0.95
    max_rescale_iter: int = 5

    # Magnitude controls
    A0_scale: float = 0.6
    Delta_scale: float = 0.3

    # Randomness
    seed: int = 123

    def __post_init__(self) -> None:
        """Normalize client indexing and sample-size inputs."""
        if self.client_T is not None:
            if not isinstance(self.client_T, dict) or len(self.client_T) == 0:
                raise ValueError("client_T must be a non-empty dict {client_id: T_k}")
            self.client_ids = sorted(int(i) for i in self.client_T.keys())
            self.K = len(self.client_ids)
            self.T_list = [int(self.client_T[i]) for i in self.client_ids]
        else:
            if self.K is None or self.T_list is None:
                raise ValueError("Provide either client_T or (K, T_list)")
            self.K = int(self.K)
            self.T_list = list(map(int, self.T_list))
            if self.K <= 0:
                raise ValueError("K must be positive")
            if len(self.T_list) != self.K:
                raise ValueError("T_list must have length K")
            self.client_ids = list(range(1, self.K + 1))
            self.client_T = {cid: self.T_list[cid - 1] for cid in self.client_ids}

        for cid, Tk in zip(self.client_ids, self.T_list):
            if Tk <= 0:
                raise ValueError(f"T_k must be positive for client {cid}, got {Tk}")


def _check_config(cfg: DGPConfig) -> None:
    """Validate a normalized DGP configuration."""
    if cfg.K is None or cfg.T_list is None:
        raise ValueError("Config not initialized: missing K/T_list")
    if cfg.K <= 0:
        raise ValueError("K must be positive")
    if cfg.d <= 0 or cfg.p <= 0:
        raise ValueError("d and p must be positive")
    if len(cfg.T_list) != cfg.K:
        raise ValueError("T_list must have length K")
    if cfg.r <= 0 or cfg.r > min(cfg.d, cfg.d * cfg.p):
        raise ValueError("r must be in [1, min(d, dp)]")
    if not (0.0 <= cfg.q < 1.0):
        raise ValueError("q must be in [0,1)")
    if cfg.s_q <= 0:
        raise ValueError("s_q must be positive")
    if cfg.enforce_stability and not (0.0 < cfg.target_rho < 1.0):
        raise ValueError("target_rho must be in (0,1) when enforce_stability=True")


# =========================
# Linear-algebra helpers
# =========================

def _companion_matrix(A_blocks: List[Array]) -> Array:
    """Build the VAR(p) companion matrix from lag blocks A_1,...,A_p."""
    p = len(A_blocks)
    d = A_blocks[0].shape[0]
    C = np.zeros((d * p, d * p), dtype=float)
    C[:d, : d * p] = np.concatenate(A_blocks, axis=1)
    if p > 1:
        C[d:, :-d] = np.eye(d * (p - 1))
    return C


def _spectral_radius(M: Array) -> float:
    """Return the spectral radius max |lambda_i(M)|."""
    vals = np.linalg.eigvals(M)
    return float(np.max(np.abs(vals)))


def _rescale_to_stable(
    A_blocks: List[Array],
    target_rho: float,
    max_iter: int = 5,
) -> Tuple[List[Array], float]:
    """Uniformly rescale lag blocks until the companion radius is below target_rho."""
    scaled = [A.copy() for A in A_blocks]
    for _ in range(max_iter):
        rho = _spectral_radius(_companion_matrix(scaled))
        if rho <= target_rho:
            return scaled, rho
        scale = target_rho / (rho + 1e-12)
        scaled = [scale * A for A in scaled]
    rho = _spectral_radius(_companion_matrix(scaled))
    return scaled, rho


def _split_A_blocks(A: Array, d: int, p: int) -> List[Array]:
    """Split A in R^{d x (dp)} into p lag blocks of shape (d,d)."""
    return [A[:, j * d : (j + 1) * d] for j in range(p)]


def _assemble_A_from_blocks(blocks: List[Array]) -> Array:
    """Concatenate lag blocks into A in R^{d x (dp)}."""
    return np.concatenate(blocks, axis=1)


# =========================
# Coefficient generation
# =========================

def _make_low_rank_A0(rng: np.random.Generator, d: int, p: int, r: int, scale: float) -> Array:
    """Generate a low-rank shared coefficient matrix A0 in R^{d x (dp)}."""
    dp = d * p
    U = rng.normal(size=(d, r))
    V = rng.normal(size=(r, dp))
    A0 = U @ V
    fro = np.linalg.norm(A0, ord="fro") + 1e-12
    return (scale / fro) * A0


def _project_to_lq_ball(Delta: Array, q: float, s_q: float) -> Array:
    """Project or rescale Delta into the l_q ball.

    For q = 0, s_q is interpreted as a sparsity budget and only the largest
    entries in magnitude are retained.
    """
    D = Delta.copy()
    if q == 0.0:
        s0 = int(round(s_q))
        if s0 <= 0:
            return np.zeros_like(D)
        flat = D.ravel()
        if s0 >= flat.size:
            return D
        idx = np.argpartition(np.abs(flat), -s0)[-s0:]
        mask = np.zeros_like(flat, dtype=bool)
        mask[idx] = True
        flat[~mask] = 0.0
        return flat.reshape(D.shape)

    val = np.sum(np.abs(D) ** q)
    if val <= s_q:
        return D
    scale = (s_q / (val + 1e-12)) ** (1.0 / q)
    return scale * D


def _make_weak_sparse_Delta(
    rng: np.random.Generator,
    d: int,
    p: int,
    q: float,
    s_q: float,
    scale: float,
    pre_sparsify: bool = True,
    sparsify_quantile: float = 0.75,
) -> Array:
    """Generate a weakly sparse client deviation Delta in R^{d x (dp)}."""
    dp = d * p
    D = rng.normal(loc=0.0, scale=1.0, size=(d, dp))

    if pre_sparsify:
        sparsify_quantile = float(sparsify_quantile)
        if not (0.0 < sparsify_quantile < 1.0):
            raise ValueError("sparsify_quantile must be in (0,1)")
        thr = np.quantile(np.abs(D), sparsify_quantile)
        D = np.sign(D) * np.maximum(np.abs(D) - 0.5 * thr, 0.0)

    fro = np.linalg.norm(D, ord="fro") + 1e-12
    D = (scale / fro) * D
    return _project_to_lq_ball(D, q=q, s_q=s_q)


def _make_Sigma_eps(rng: np.random.Generator, cfg: DGPConfig) -> List[Array]:
    """Build one innovation covariance matrix per client."""
    d = cfg.d
    if cfg.Sigma_eps is None:
        return [np.eye(d) for _ in range(cfg.K)]

    if isinstance(cfg.Sigma_eps, str):
        if cfg.Sigma_eps.lower() == "diag":
            out = []
            for _ in range(cfg.K):
                diag = rng.uniform(0.5, 1.5, size=d)
                out.append(np.diag(diag))
            return out
        raise ValueError("Sigma_eps string must be None or 'diag'")

    if len(cfg.Sigma_eps) != cfg.K:
        raise ValueError("Sigma_eps list must have length K")

    out = []
    for S in cfg.Sigma_eps:
        if S.shape != (d, d):
            raise ValueError("Each Sigma_eps[k] must be (d,d)")
        S = 0.5 * (S + S.T)
        eigmin = np.min(np.linalg.eigvalsh(S))
        if eigmin < 1e-8:
            S = S + (1e-8 - eigmin) * np.eye(d)
        out.append(S)
    return out


# =========================
# Simulation
# =========================

def simulate_var_client(
    rng: np.random.Generator,
    A_blocks: List[Array],
    Sigma_eps: Array,
    T: int,
    burn_in: int,
    y0: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Simulate one client-specific VAR(p) trajectory."""
    p = len(A_blocks)
    d = A_blocks[0].shape[0]
    total_T = T + burn_in

    L = np.linalg.cholesky(Sigma_eps)
    eps_full = rng.normal(size=(total_T, d)) @ L.T

    if y0 is None:
        lag = np.zeros((p, d), dtype=float)
    else:
        y0 = np.asarray(y0, dtype=float)
        if y0.shape != (p, d):
            raise ValueError("y0 must have shape (p,d)")
        lag = y0.copy()

    y_full = np.zeros((total_T, d), dtype=float)
    for t in range(total_T):
        y_t = np.zeros(d, dtype=float)
        for j in range(p):
            y_t += A_blocks[j] @ lag[j]
        y_t += eps_full[t]
        y_full[t] = y_t

        if p > 1:
            lag[1:] = lag[:-1]
        lag[0] = y_t

    return y_full[burn_in:], eps_full[burn_in:]


# =========================
# Public API
# =========================

def generate_coefficients(cfg: DGPConfig) -> Dict[str, object]:
    """Generate only the coefficient objects for the multi-client VAR(p) DGP.

    Returns a dictionary containing:
    - `A0`: shared low-rank coefficient matrix in R^{d x (dp)};
    - `Delta`: list of client-specific deviations;
    - `A`: list of full client-specific coefficient matrices;
    - `A_blocks`: list of lag-block representations for simulation;
    - `Sigma_eps`: list of innovation covariance matrices;
    - `spectral_radius`: list of client-specific companion radii.
    """
    _check_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    d, p, K = cfg.d, cfg.p, cfg.K
    A0 = _make_low_rank_A0(rng, d=d, p=p, r=cfg.r, scale=cfg.A0_scale)
    A0_blocks = _split_A_blocks(A0, d=d, p=p)

    if cfg.enforce_stability:
        A0_blocks, rho0 = _rescale_to_stable(
            A0_blocks,
            target_rho=cfg.target_rho,
            max_iter=cfg.max_rescale_iter,
        )
    else:
        rho0 = _spectral_radius(_companion_matrix(A0_blocks))
    A0 = _assemble_A_from_blocks(A0_blocks)

    Delta_list: List[Array] = []
    A_list: List[Array] = []
    A_blocks_list: List[List[Array]] = []
    rho_list: List[float] = []
    alpha_list: List[float] = []

    Sigma_list = _make_Sigma_eps(rng, cfg)

    for _ in range(K):
        Delta_k = _make_weak_sparse_Delta(
            rng,
            d=d,
            p=p,
            q=cfg.q,
            s_q=cfg.s_q,
            scale=cfg.Delta_scale,
            pre_sparsify=True,
            sparsify_quantile=0.75,
        )

        Delta_blocks = _split_A_blocks(Delta_k, d=d, p=p)
        blocks = [A0_blocks[j] + Delta_blocks[j] for j in range(p)]
        rho = _spectral_radius(_companion_matrix(blocks))

        alpha = 1.0
        if cfg.enforce_stability and rho > cfg.target_rho:
            alpha = float(cfg.target_rho / (rho + 1e-12))
            blocks = [A0_blocks[j] + alpha * Delta_blocks[j] for j in range(p)]
            rho = _spectral_radius(_companion_matrix(blocks))

        A_k = _assemble_A_from_blocks(blocks)
        Delta_list.append(A_k - A0)
        A_list.append(A_k)
        A_blocks_list.append(blocks)
        rho_list.append(rho)
        alpha_list.append(alpha)

    return {
        "A0": A0,
        "Delta": Delta_list,
        "A": A_list,
        "A_blocks": A_blocks_list,
        "Sigma_eps": Sigma_list,
        "spectral_radius": rho_list,
        "config": cfg,
        "seed_used": int(cfg.seed),
        "rho_A0": float(rho0),
        "alpha": alpha_list,
    }


def simulate_from_coefficients(
    coeffs: Dict[str, object],
    T_list: Sequence[int],
    burn_in: int = 200,
    seed: int = 123,
) -> Dict[str, object]:
    """Simulate multi-client time series from pre-generated coefficients."""
    A_blocks_list = coeffs["A_blocks"]
    Sigma_list = coeffs["Sigma_eps"]

    K = len(A_blocks_list)
    if len(T_list) != K:
        raise ValueError("T_list must have length K to match coefficients")

    rng = np.random.default_rng(seed)
    y_list: List[Array] = []
    eps_list: List[Array] = []

    for k in range(K):
        yk, ek = simulate_var_client(
            rng=rng,
            A_blocks=A_blocks_list[k],
            Sigma_eps=Sigma_list[k],
            T=int(T_list[k]),
            burn_in=int(burn_in),
            y0=None,
        )
        y_list.append(yk)
        eps_list.append(ek)

    return {
        "y": y_list,
        "eps": eps_list,
        "T_list": list(map(int, T_list)),
        "burn_in": int(burn_in),
        "seed_used": int(seed),
    }


def generate_dgp(cfg: DGPConfig) -> Dict[str, object]:
    """Generate coefficients and simulated data in one call."""
    coeffs = generate_coefficients(cfg)
    sim = simulate_from_coefficients(
        coeffs,
        T_list=cfg.T_list,
        burn_in=cfg.burn_in,
        seed=cfg.seed + 999,
    )
    out = dict(coeffs)
    out.update(sim)
    return out