"""EuRoC reader and run script on a small fake sequence in the ASL format (made from a UAV scenario)."""

import numpy as np
import pytest

from experiments.nav_compare.data.euroc import euroc_recording
from experiments.nav_compare.data.scenarios import generate, scenario_grid
from experiments.nav_compare.scripts import run_euroc


def write_fake_euroc(folder, duration=20.0):
    seq = generate(
        scenario_grid("uav")["baseline"].replace(outage_s=(), duration_s=duration),
        seed=4,
    )
    ref, tr = seq.reference, seq.truth
    t_ns = (1_403_636_579_000_000_000 + np.round(seq.imu.t_s * 1e9)).astype(np.int64)
    imu = folder / "mav0" / "imu0"
    gt = folder / "mav0" / "state_groundtruth_estimate0"
    imu.mkdir(parents=True)
    gt.mkdir(parents=True)
    rows = np.column_stack(
        [t_ns.astype(object), seq.imu.gyro_rad_s, seq.imu.accel_m_s2]
    )
    fmt = ["%d"] + ["%.12g"] * 6
    np.savetxt(imu / "data.csv", rows, delimiter=",", header="timestamp,w,a", fmt=fmt)
    rows = np.column_stack([t_ns.astype(object), ref.p_w, ref.q_wb, ref.v_w, tr["gyro_bias_series"],
                            tr["accel_bias_series"]])  # fmt: skip
    np.savetxt(
        gt / "data.csv",
        rows,
        delimiter=",",
        header="timestamp,p,q,v,bw,ba",
        fmt=["%d"] + ["%.12g"] * 16,
    )
    return seq


def test_reader_matches_source(tmp_path):
    folder = tmp_path / "V1_01_easy"
    seq = write_fake_euroc(folder)
    rec = euroc_recording(folder, pos_rate_hz=10.0, pos_std_m=0.1, outage=(5.0, 8.0))
    assert np.allclose(rec["truth_p"], seq.reference.p_w, atol=1e-6)
    assert np.allclose(rec["imu_gyro"], seq.imu.gyro_rad_s, atol=1e-8)
    assert rec["pos_valid"].sum() < len(rec["pos_valid"])
    assert rec["events"][0]["type"] == "outage"


def test_locked_sequence_is_refused(tmp_path):
    with pytest.raises(PermissionError):
        euroc_recording(tmp_path / "V1_03_difficult")


def test_run_script(tmp_path):
    folder = tmp_path / "MH_01_easy"
    write_fake_euroc(folder, duration=30.0)
    out = tmp_path / "out"
    run_euroc.main([str(folder), "--filters", "KF-CV", "ESKF", "--out", str(out)])
    text = (out / "runs.csv").read_text()
    assert "ESKF" in text and "outage" in text
    assert len(list((out / "curves").glob("*.npz"))) == 2
    from experiments.nav_compare import peaks

    peaks.main(["curves", str(out / "curves"), "--out", str(tmp_path / "pk")])
    assert (tmp_path / "pk" / "condition_table.csv").exists()
