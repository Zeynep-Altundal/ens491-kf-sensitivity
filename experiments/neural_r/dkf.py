"""Differentiable Kalman filter (PyTorch, float64) for the linear model, and the learned-R models.

The filter follows ``experiments/linear_kf/kf.py`` step by step (prediction ``x <- F x``,
``P <- F P F^T + Q``; update with ``K = (S^-1 H P)^T`` and the Joseph-form covariance), batched
over trajectories, with a measurement covariance that may change per step and per trajectory.

Learned covariance (``README.md``, Part 1)::

    R_t = diag(alpha_t) R_base,   alpha_t = softplus(f_theta(features of steps 1..t-1)) + eps

Timing at step t: prediction; ``alpha_t`` from the features of completed steps only; innovation
and ``S_t``; update; features of step t.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64
EPS = 1e-3
GRU_HIDDEN = 16
MLP_WINDOW = 20
MLP_HIDDEN = 32
INIT_BIAS = math.log(math.e - 1.0)  # softplus(INIT_BIAS) = 1


def features(nu: torch.Tensor, S: torch.Tensor, R_base_diag: torch.Tensor) -> torch.Tensor:
    """Per-step features (B, 3m): nu / sqrt(R_base), nu / sqrt(S_ii), ln(S_ii / R_base)."""
    s = torch.diagonal(S, dim1=-2, dim2=-1)
    return torch.cat([nu / torch.sqrt(R_base_diag), nu / torch.sqrt(s),
                      torch.log(s / R_base_diag)], dim=-1)


class GRUCovariance(nn.Module):
    """One GRU layer (hidden 16) and a linear output layer; input at step t: features of t-1."""

    kind = "gru"

    def __init__(self, m: int):
        super().__init__()
        self.m = m
        self.cell = nn.GRUCell(3 * m, GRU_HIDDEN, dtype=DTYPE)
        self.head = nn.Linear(GRU_HIDDEN, m, dtype=DTYPE)
        with torch.no_grad():
            self.head.bias.fill_(INIT_BIAS)

    def init_state(self, batch: int):
        return {"h": torch.zeros(batch, GRU_HIDDEN, dtype=DTYPE),
                "f": torch.zeros(batch, 3 * self.m, dtype=DTYPE)}

    def alpha(self, state):
        state["h"] = self.cell(state["f"], state["h"])
        return nn.functional.softplus(self.head(state["h"])) + EPS

    def observe(self, state, feat):
        state["f"] = feat


class MLPCovariance(nn.Module):
    """Two tanh layers of 32 units on the flattened features of the last 20 steps (zero-padded)."""

    kind = "mlp"

    def __init__(self, m: int):
        super().__init__()
        self.m = m
        self.net = nn.Sequential(
            nn.Linear(MLP_WINDOW * 3 * m, MLP_HIDDEN, dtype=DTYPE), nn.Tanh(),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN, dtype=DTYPE), nn.Tanh(),
            nn.Linear(MLP_HIDDEN, m, dtype=DTYPE))
        with torch.no_grad():
            self.net[-1].bias.fill_(INIT_BIAS)

    def init_state(self, batch: int):
        return {"buf": torch.zeros(batch, MLP_WINDOW, 3 * self.m, dtype=DTYPE)}

    def alpha(self, state):
        return nn.functional.softplus(self.net(state["buf"].flatten(1))) + EPS

    def observe(self, state, feat):
        state["buf"] = torch.cat([state["buf"][:, 1:], feat[:, None]], dim=1)


MODELS = {"gru": GRUCovariance, "mlp": MLPCovariance}


def as_tensor(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=float), dtype=DTYPE)


def run_kf(model, y: torch.Tensor, net: nn.Module | None = None,
           R_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Run the filter on y (B, T, m).

    The measurement covariance is, in order of precedence, the learned ``net``, the sequence
    ``R_seq`` (B, T, m, m), or the model's R. ``model`` is a ``StateSpaceModel`` of a linear model.
    Returns posterior means x (B, T, n), covariances P (B, T, n, n), innovations nu (B, T, m),
    innovation covariances S (B, T, m, m) and the R used (B, T, m, m).
    """
    B, T, m = y.shape
    F = as_tensor(model.F_jacobian(None))
    H = as_tensor(model.H_jacobian(None))
    Q = as_tensor(model.Q)
    R_base = as_tensor(model.R)
    R_base_diag = torch.diagonal(R_base)
    n = F.shape[0]
    eye = torch.eye(n, dtype=DTYPE)
    x = as_tensor(model.x0).expand(B, n)
    P = as_tensor(model.P0).expand(B, n, n)
    state = net.init_state(B) if net is not None else None
    out = {k: [] for k in ("x", "P", "nu", "S", "R")}
    for t in range(T):
        x = x @ F.T
        P = F @ P @ F.T + Q
        if net is not None:
            R = torch.diag_embed(net.alpha(state) * R_base_diag)
        elif R_seq is not None:
            R = R_seq[:, t]
        else:
            R = R_base.expand(B, m, m)
        S = H @ P @ H.T + R
        K = torch.linalg.solve(S, H @ P).transpose(-1, -2)
        nu = y[:, t] - x @ H.T
        x = x + (K @ nu[..., None])[..., 0]
        A = eye - K @ H
        P = A @ P @ A.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
        if net is not None:
            net.observe(state, features(nu, S, R_base_diag))
        for k, v in (("x", x), ("P", P), ("nu", nu), ("S", S), ("R", R)):
            out[k].append(v)
    return {k: torch.stack(v, dim=1) for k, v in out.items()}


def innovation_nll(nu: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Per-step NLL 0.5 (nu^T S^-1 nu + ln det S + m ln 2 pi), shape (B, T)."""
    m = nu.shape[-1]
    sol = torch.linalg.solve(S, nu[..., None])[..., 0]
    return 0.5 * ((nu * sol).sum(-1) + torch.logdet(S) + m * math.log(2.0 * math.pi))
