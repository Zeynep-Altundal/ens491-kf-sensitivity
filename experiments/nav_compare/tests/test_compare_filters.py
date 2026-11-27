import numpy as np
import pytest

from experiments.nav_compare.compare.filters import FILTERS
from experiments.nav_compare.compare.runner import errors, run_filter, summarize
from experiments.nav_compare.data.disturbances import apply_events, standard_event_sets
from experiments.nav_compare.data.recording import to_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid


def _base(platform, **kw):
    kw = {"outage_s": (), "duration_s": 30.0, **kw}
    sc = scenario_grid(platform)["baseline"].replace(**kw)
    return generate(sc, seed=5)


@pytest.fixture(scope="module", params=["ugv", "uav"])
def clean_rec(request):
    return to_recording(_base(request.param))


@pytest.fixture(scope="module", params=["ugv", "uav"])
def ideal_rec(request):
    # (almost) no IMU error, no bias, very accurate position: the IMU filters must follow the truth
    seq = _base(request.param, noise_scale=1e-6, bias_scale=0.0, bias_random_walk=False,
                obs_std_m=1e-3)  # fmt: skip
    return to_recording(seq)


def test_log_layout_and_nan_policy(clean_rec):
    for name in FILTERS:
        log = run_filter(clean_rec, name, log_every=20)
        n = len(log["t"])
        assert log["p_hat"].shape == (n, 3) and log["P_pp"].shape == (n, 3, 3)
        assert len(log["nis"]) == len(log["upd_t"]) > 0
        if name == "KF-CV":
            assert np.all(np.isnan(log["q_hat"])) and np.all(np.isnan(log["ba_hat"]))
        if name == "KF-IMU":
            assert np.all(np.isnan(log["q_hat"])) and np.all(np.isfinite(log["ba_hat"]))
        if name in ("EKF", "ESKF"):
            assert np.all(np.isfinite(log["q_hat"])) and np.all(
                np.isfinite(log["P_thth"])
            )


def test_imu_filters_follow_truth_without_errors(ideal_rec):
    for name in ("KF-IMU", "EKF", "ESKF"):
        s = summarize(ideal_rec, run_filter(ideal_rec, name), t_skip=2.0)
        assert s["max_p"] < 0.05, (name, s["max_p"])


def test_ekf_and_eskf_agree_on_clean_data(clean_rec):
    a = summarize(clean_rec, run_filter(clean_rec, "EKF"))
    b = summarize(clean_rec, run_filter(clean_rec, "ESKF"))
    assert abs(a["rmse_p"] / b["rmse_p"] - 1.0) < 0.02
    assert abs(a["rmse_att_deg"] / b["rmse_att_deg"] - 1.0) < 0.05


def test_imu_filters_beat_position_only_kf(clean_rec):
    r = {n: summarize(clean_rec, run_filter(clean_rec, n))["rmse_p"] for n in FILTERS}
    assert max(r["KF-IMU"], r["ESKF"], r["EKF"]) < r["KF-CV"]


def test_kf_imu_estimates_a_constant_accel_bias():
    seq = _base("uav", bias_scale=5.0, bias_random_walk=False)
    rec = to_recording(seq)
    E = errors(rec, run_filter(rec, "KF-IMU"))
    true_ba = rec["truth_ba"][-1]
    assert np.linalg.norm(E["e_ba"][-1]) < 0.3 * np.linalg.norm(true_ba)


def test_event_metrics_present():
    seq = _base("ugv", duration_s=60.0)
    evs = standard_event_sets("ugv", 60.0)["pos_noise_burst"]
    rec = to_recording(apply_events(seq, evs, seed=1))
    s = summarize(rec, run_filter(rec, "ESKF"))
    assert {"pre_err", "peak_err", "peak_t", "recovery_s"} <= set(s)
    assert s["peak_err"] > s["pre_err"]
