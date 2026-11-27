"""Fast tests for experiments.nav_compare.data.scenarios: truth consistency, determinism, paired seeds."""

import numpy as np
import pytest

from experiments.nav_compare.baselines.strapdown import (
    oracle_initial_state,
    run_strapdown,
)
from experiments.nav_compare.data.scenarios import (
    Scenario,
    generate,
    initial_state_for,
    scenario_grid,
)
from experiments.nav_compare.data.sequence import split_into_segments

CLEAN = {"noise_scale": 0.0, "bias_scale": 0.0, "bias_random_walk": False}


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"slip_rad": 0.05, "grade_rad": 0.08},  # UGV only, ignored by UAV
        {"uav_period_s": 8.0},  # UAV only, ignored by UGV
    ],
)
@pytest.mark.parametrize("platform", ["ugv", "uav"])
def test_clean_scenario_strapdown_matches_truth(platform, extra):
    sc = Scenario(platform=platform, duration_s=30.0, **CLEAN, **extra)
    seq = generate(sc, seed=0)
    p, _, _ = run_strapdown(seq.imu, oracle_initial_state(seq.reference))
    err = np.linalg.norm(p - seq.reference.p_w, axis=1).max()
    assert err < 0.2, err


def test_deterministic_and_paired_streams():
    a = generate(Scenario(platform="uav", duration_s=10.0), seed=3)
    b = generate(Scenario(platform="uav", duration_s=10.0), seed=3)
    assert a.source_hash == b.source_hash
    # a different factor must not reshuffle the white noise or the bias of the others
    c = generate(Scenario(platform="uav", duration_s=10.0, vibration_accel=0.3), seed=3)
    assert np.allclose(a.truth["accel_bias_true"], c.truth["accel_bias_true"])
    diff = c.imu.accel_m_s2 - a.imu.accel_m_s2
    assert np.abs(diff).max() <= 0.3 + 1e-9 and np.abs(diff).max() > 0.1


def test_dropout_creates_gaps_and_keeps_alignment():
    sc = Scenario(
        platform="ugv", duration_s=60.0, dropout_bursts_per_min=6.0, dropout_burst_s=0.2
    )
    seq = generate(sc, seed=1)
    assert seq.truth["dropped_samples"] > 0
    assert len(seq.imu.t_s) == len(seq.reference.t_s)
    _, gaps = split_into_segments(seq.imu.t_s, seq.imu.gyro_rad_s, seq.imu.accel_m_s2)
    assert gaps >= 1


def test_observation_defects():
    sc = Scenario(
        platform="ugv", duration_s=120.0, obs_outlier_prob=0.5, obs_assumed_factor=2.0
    )
    seq = generate(sc, seed=0)
    assert any(o.provenance["outlier"] for o in seq.observations)
    assert all(
        np.allclose(o.r_meas, np.eye(3) * (0.5 * 2.0) ** 2) for o in seq.observations
    )
    assert not any(60.0 <= o.t_s < 90.0 for o in seq.observations)
    full = generate(Scenario(platform="ugv", outage_s=()), seed=0)
    assert full.hidden_observations == 0


def test_initial_state_perturbation():
    sc = Scenario(
        platform="ugv", duration_s=10.0, init_att_std_rad=0.05, init_bias_known=False
    )
    seq = generate(sc, seed=0)
    s = initial_state_for(sc, seq, seed=0)
    assert s.init == "operational" and np.allclose(s.b_g, 0.0)
    assert not np.allclose(s.q, seq.reference.q_wb[0])
    clean = initial_state_for(Scenario(platform="ugv"), seq, seed=0)
    assert clean.init == "oracle"


@pytest.mark.parametrize("platform", ["ugv", "uav"])
def test_grid_builds_and_every_scenario_runs(platform):
    grid = scenario_grid(platform)
    assert "baseline" in grid and len(grid) > 30
    for name, sc in grid.items():
        seq = generate(sc.replace(duration_s=15.0, outage_s=(5.0, 8.0)), seed=0)
        assert np.all(np.isfinite(seq.imu.accel_m_s2)), name
