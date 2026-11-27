import json

import numpy as np

from experiments.nav_compare import peaks


def test_detect_single_peak():
    rng = np.random.default_rng(0)
    e = 1.0 + 0.05 * rng.standard_normal(500)
    e[200:210] += 3.0
    pk, thr = peaks.detect_peaks(e)
    assert len(pk) == 1
    i0, i1, im = pk[0]
    assert 199 <= i0 <= 201 and 208 <= i1 <= 210 and 200 <= im <= 209
    assert 1.5 < thr < 3.0


def test_unique_and_why():
    rng = np.random.default_rng(1)
    base = 1.0 + 0.05 * rng.standard_normal(300)
    a, b = base.copy(), base.copy()
    a[100:105] += 4.0  # both filters
    b[100:104] += 4.0
    b[250:255] += 4.0  # only b
    nees = np.full(300, 3.0)
    nees_b = nees.copy()
    nees_b[250:255] = 100.0
    rows = peaks.analyse(
        {"A": {"e": a, "nees": nees, "nis": None}, "B": {"e": b, "nees": nees_b, "nis": None},
         "Net": {"e": base, "nees": np.full(300, np.nan), "nis": None}}, m=3, n=3)  # fmt: skip
    by = {(r["filter"], r["i0"] // 50): r for r in rows}
    assert not by[("A", 2)]["unique"] and not by[("B", 2)]["unique"]
    assert by[("B", 5)]["unique"]
    assert by[("B", 5)]["why"] == "overconfident"
    assert by[("A", 2)]["why"] == "aware"


def test_knet_format(tmp_path):
    N, T, m, n = 3, 80, 2, 2
    rng = np.random.default_rng(2)
    meta = json.dumps({"source": "kalmannet_linear", "state_names": ["p", "v"]})
    folder = tmp_path / "linear_full_F0"
    folder.mkdir()
    for f in ("KF", "KalmanNet"):
        err = 0.1 * rng.standard_normal((N, T, m)).astype(np.float32)
        err[1, 40:43] += 5.0
        nan = f == "KalmanNet"
        np.savez(folder / f"{f}.npz", t=np.arange(1, T + 1, dtype=float), x_hat=err, err=err,
                 nees=np.full((N, T), np.nan if nan else 2.0, np.float32),
                 innov=np.zeros((N, T, n), np.float32),
                 nis=np.full((N, T), np.nan if nan else 2.0, np.float32), meta=meta)  # fmt: skip
    runs = list(peaks.knet_runs(tmp_path))
    assert len(runs) == N
    out = tmp_path / "out"
    peaks.main(["knet", str(tmp_path), "--out", str(out)])
    text = (out / "peaks.csv").read_text()
    assert "KalmanNet" in text and "no covariance" in text
