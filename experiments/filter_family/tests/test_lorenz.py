"""Tests of the Lorenz-attractor model used for the KalmanNet benchmark."""

import numpy as np
from scipy.linalg import expm

from experiments.filter_family.lorenz import (
    BETA,
    DT,
    RHO,
    SIGMA,
    jacobian_exact,
    lorenz_A,
    simulate,
    taylor_F,
)


def _field(x):
    return np.array([SIGMA * (x[1] - x[0]), x[0] * (RHO - x[2]) - x[1], x[0] * x[1] - BETA * x[2]])


def test_both_forms_of_A_give_the_lorenz_vector_field_but_different_taylor_terms():
    rng = np.random.default_rng(0)
    for x in rng.normal(0.0, 10.0, (20, 3)):
        assert np.allclose(lorenz_A(x, "code") @ x, _field(x), rtol=1e-13, atol=1e-12)
        assert np.allclose(lorenz_A(x, "paper") @ x, _field(x), rtol=1e-13, atol=1e-12)
    x = np.array([3.0, -2.0, 20.0])
    d = taylor_F(x, 5, form="code") @ x - taylor_F(x, 5, form="paper") @ x
    print("difference of one J = 5 step between the two forms:", d)
    assert np.linalg.norm(d) > 1e-6  # the forms define different discrete models


def test_taylor_sum_converges_to_matrix_exponential():
    x = np.array([1.0, 2.0, 25.0])
    assert np.allclose(taylor_F(x, 25), expm(lorenz_A(x) * DT), rtol=1e-12, atol=1e-14)


def test_exact_jacobian_matches_analytic_for_first_order():
    x = np.array([2.0, -1.0, 15.0])
    JL = np.array([[-SIGMA, SIGMA, 0.0], [RHO - x[2], -1.0, -x[0]], [x[1], x[0], -BETA]])
    assert np.allclose(jacobian_exact(1)(x), np.eye(3) + DT * JL, rtol=1e-8, atol=1e-8)


def test_simulation_is_seeded_and_excludes_x0():
    a = simulate(0.1, 3, 50, np.random.default_rng(7))
    b = simulate(0.1, 3, 50, np.random.default_rng(7))
    assert a[0].shape == (3, 50, 3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
