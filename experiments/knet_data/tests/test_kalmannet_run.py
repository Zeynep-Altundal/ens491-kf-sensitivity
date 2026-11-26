"""Tests for the torch model functions given to KalmanNet (need PyTorch, not the reference code).

Run from the repository root with ``python -m pytest -v experiments/knet_data``.
"""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.filter_family import lorenz as zl  # noqa: E402
from experiments.knet_data import kalmannet_run as kr, sinusoidal as sin  # noqa: E402


def test_lorenz_torch_f_matches_numpy():
    x = np.random.default_rng(0).normal(size=(5, 3)) * 5
    for J in (zl.J_DATA, zl.J_MISMATCH):
        ours = kr.lorenz_f(J, torch.tensor(x[:, :, None], dtype=torch.float64))[:, :, 0].numpy()
        np.testing.assert_allclose(ours, zl.make_f(J)(x), rtol=1e-10)


def test_sinusoidal_torch_functions_match_numpy():
    x = np.random.default_rng(1).normal(size=(4, 2))
    xt = torch.tensor(x[:, :, None])
    for p in sin.PARAMS.values():
        np.testing.assert_allclose(kr.sin_f(p, xt)[:, :, 0].numpy(), sin.f(x, p), rtol=1e-12)
        np.testing.assert_allclose(kr.sin_h(p, xt)[:, :, 0].numpy(), sin.h(x, p), rtol=1e-12)


def test_model_functions_are_picklable():
    # The reference pipeline saves the whole network (including f and h) with torch.save.
    arrays = {"F_model": np.eye(2), "H_model": np.eye(2)}
    f, h, _ = kr.torch_model(arrays, {"benchmark": "linear"}, "model")
    pickle.dumps((f, h))