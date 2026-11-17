"""Run the ESKF on a synthetic scenario.

For run index ``i`` the scenario uses ``seed = i`` and draws the turn-on biases
and the initial state error from the filter's initial covariance P0 (generator
seeded with ``[INIT_SEED, i]``), so that the initial uncertainty given to the
filter is consistent with the actual initial error. The filter is given the
true IMU noise parameters and the true measurement covariance.

Usage (from the repository root)::

    python -m experiments.eskf_sensitivity.run
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthetic.rotation import quat_exp, quat_multiply, quat_normalize
from synthetic.scenario import ScenarioConfig, generate

from .eskf import (
    IDX_P,
    ErrorStateKalmanFilter,
    ImuNoise,
    NominalState,
    difference,
)

INIT_SEED = 1000
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_PROFILE = "turning"
DEFAULT_RUN_INDEX = 0


@dataclass(frozen=True)
class FilterSettings:
    """Initial standard deviations (per axis) and the optional innovation gate."""

    sigma_p0: float = 0.5  # [m]
    sigma_v0: float = 0.1  # [m/s]
    sigma_theta0: float = 0.01  # [rad]
    sigma_bg0: float = 2.0e-3  # [rad/s]
    sigma_ba0: float = 0.05  # [m/s^2]
    gate_prob: float | None = None

    def P0(self) -> np.ndarray:
        sig = np.repeat([self.sigma_p0, self.sigma_v0, self.sigma_theta0,
                         self.sigma_bg0, self.sigma_ba0], 3)
        return np.diag(sig**2)


def scenario_config(profile: str, run_index: int, settings: FilterSettings,
                    **overrides) -> ScenarioConfig:
    """Scenario with turn-on biases drawn from the filter's initial bias covariance."""
    rng = np.random.default_rng([INIT_SEED, run_index])
    bg = rng.normal(0.0, settings.sigma_bg0, 3)
    ba = rng.normal(0.0, settings.sigma_ba0, 3)
    return ScenarioConfig(profile=profile, seed=run_index,
                          gyro_turn_on_bias=tuple(float(b) for b in bg),
                          accel_turn_on_bias=tuple(float(b) for b in ba), **overrides)


def initial_state(out: dict, settings: FilterSettings, run_index: int) -> NominalState:
    """Initial nominal state: truth minus an error drawn from P0; biases start at zero.

    The bias part of the initial error equals the true turn-on bias, which was
    drawn from the same P0 block in :func:`scenario_config`.
    """
    rng = np.random.default_rng([INIT_SEED, run_index, 1])
    dp = rng.normal(0.0, settings.sigma_p0, 3)
    dv = rng.normal(0.0, settings.sigma_v0, 3)
    dth = rng.normal(0.0, settings.sigma_theta0, 3)
    # q_true = q ⊗ Exp(dth)  =>  q = q_true ⊗ Exp(-dth)
    q = quat_normalize(quat_multiply(out["q_true"][0], quat_exp(-dth)))
    return NominalState(p=out["p_true"][0] - dp, v=out["v_true"][0] - dv, q=q,
                        bg=np.zeros(3), ba=np.zeros(3))


def exact_initial_state(out: dict) -> NominalState:
    """Initial nominal state equal to the truth (zero biases)."""
    return NominalState(p=out["p_true"][0].copy(), v=out["v_true"][0].copy(),
                        q=out["q_true"][0].copy(), bg=np.zeros(3), ba=np.zeros(3))


def imu_noise(cfg: ScenarioConfig) -> ImuNoise:
    return ImuNoise(cfg.gyro_noise_density, cfg.accel_noise_density,
                    cfg.gyro_bias_rw, cfg.accel_bias_rw)


def run_filter(out: dict, cfg: ScenarioConfig, settings: FilterSettings,
               x0: NominalState, *, noise: ImuNoise | None = None, R: np.ndarray | None = None,
               P0: np.ndarray | None = None,
               eval_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run the filter over a generated scenario.

    By default the filter uses the true IMU noise (``imu_noise(cfg)``), the
    true measurement covariance and ``settings.P0()``; ``noise``, ``R`` and
    ``P0`` override these filter values. The history is recorded at the IMU
    sample indices ``eval_index`` (default: the measurement epochs), after the
    update when a measurement is processed at that sample and after
    propagation otherwise. NIS and acceptance flags are recorded per
    measurement epoch. Also returns the maximum position and attitude errors
    over all IMU samples.
    """
    t, gyro, accel = out["t"], out["gyro_meas"], out["accel_meas"]
    if R is None:
        R = np.diag(np.asarray(cfg.meas_sigma, dtype=float) ** 2)
    meas_row = {int(k): j for j, k in enumerate(out["meas_index"])}
    if eval_index is None:
        eval_index = out["meas_index"]
    eval_row = {int(k): j for j, k in enumerate(eval_index)}

    kf = ErrorStateKalmanFilter(x0, settings.P0() if P0 is None else P0,
                                imu_noise(cfg) if noise is None else noise, gyro[0], accel[0],
                                gate_prob=settings.gate_prob)
    m = len(out["meas_index"])
    e = len(eval_index)
    hist = {
        "t": t[eval_index].copy(),
        "err": np.empty((e, 15)),
        "sigma": np.empty((e, 15)),
        "p_est": np.empty((e, 3)),
        "bg_est": np.empty((e, 3)),
        "ba_est": np.empty((e, 3)),
        "P_pos": np.empty((e, 3, 3)),
        "nis": np.full(m, np.nan),
        "accepted": np.zeros(m, dtype=bool),
        "available": out["meas_available"].copy(),
    }
    max_pos_err = 0.0
    max_att_err = 0.0
    for k in range(len(t)):
        if k > 0:
            kf.propagate(gyro[k], accel[k], t[k] - t[k - 1])
        x_true = NominalState(out["p_true"][k], out["v_true"][k], out["q_true"][k],
                              out["gyro_bias"][k], out["accel_bias"][k])
        if k in meas_row:
            j = meas_row[k]
            if out["meas_available"][j]:
                res = kf.update_position(out["z_meas"][j], R)
                hist["nis"][j] = res.nis
                hist["accepted"][j] = res.accepted
        if k in eval_row:
            j = eval_row[k]
            x = kf.state
            hist["err"][j] = difference(x_true, x)
            hist["sigma"][j] = np.sqrt(np.diag(kf.P))
            hist["p_est"][j] = x.p
            hist["bg_est"][j] = x.bg
            hist["ba_est"][j] = x.ba
            hist["P_pos"][j] = kf.P[IDX_P, IDX_P]
        dx = difference(x_true, kf.state)
        max_pos_err = max(max_pos_err, float(np.linalg.norm(dx[0:3])))
        max_att_err = max(max_att_err, float(np.linalg.norm(dx[6:9])))

    err_p = hist["err"][:, 0:3]
    hist["nees_pos"] = np.einsum("ti,tij,tj->t", err_p, np.linalg.inv(hist["P_pos"]), err_p)
    hist["max_pos_err"] = max_pos_err
    hist["max_att_err"] = max_att_err
    return hist


def summarize(hist: dict) -> dict[str, float]:
    e = hist["err"]
    return {
        "pos_rmse_3d": float(np.sqrt(np.mean(np.sum(e[:, 0:3] ** 2, axis=1)))),
        "vel_rmse_3d": float(np.sqrt(np.mean(np.sum(e[:, 3:6] ** 2, axis=1)))),
        "att_rmse_3d_deg": float(np.rad2deg(np.sqrt(np.mean(np.sum(e[:, 6:9] ** 2, axis=1))))),
        "mean_nees_pos": float(np.mean(hist["nees_pos"])),
        "mean_nis": float(np.nanmean(hist["nis"])),
        "n_updates": int(np.sum(hist["accepted"])),
        "max_pos_err": hist["max_pos_err"],
        "max_att_err_deg": float(np.rad2deg(hist["max_att_err"])),
    }


STATE_LABELS = [f"{blk}_{ax}" for blk in ("p", "v", "th", "bg", "ba") for ax in "xyz"]


def write_history_csv(hist: dict, out: dict, path: Path) -> None:
    """One row per measurement epoch."""
    idx = out["meas_index"]
    header = (["t"] + [f"p_true_{a}" for a in "xyz"] + [f"z_{a}" for a in "xyz"]
              + [f"p_est_{a}" for a in "xyz"]
              + [f"err_{s}" for s in STATE_LABELS] + [f"sigma_{s}" for s in STATE_LABELS]
              + [f"bg_true_{a}" for a in "xyz"] + [f"bg_est_{a}" for a in "xyz"]
              + [f"ba_true_{a}" for a in "xyz"] + [f"ba_est_{a}" for a in "xyz"]
              + ["nis", "nees_pos", "available", "accepted"])
    rows = np.column_stack([
        hist["t"], out["p_true"][idx], out["z_meas"], hist["p_est"],
        hist["err"], hist["sigma"],
        out["gyro_bias"][idx], hist["bg_est"], out["accel_bias"][idx], hist["ba_est"],
        hist["nis"], hist["nees_pos"], hist["available"], hist["accepted"],
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="", fmt="%.10g")


def write_summary_csv(summary: dict, cfg: ScenarioConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["profile", "seed"] + list(summary))
        writer.writerow([cfg.profile, cfg.seed]
                        + [f"{v:.10g}" if isinstance(v, float) else v for v in summary.values()])


def main() -> None:
    settings = FilterSettings()
    cfg = scenario_config(DEFAULT_PROFILE, DEFAULT_RUN_INDEX, settings)
    out = generate(cfg)
    hist = run_filter(out, cfg, settings, initial_state(out, settings, DEFAULT_RUN_INDEX))
    summary = summarize(hist)
    stem = f"{DEFAULT_PROFILE}_run{DEFAULT_RUN_INDEX}"
    write_history_csv(hist, out, RESULTS_DIR / f"{stem}_history.csv")
    write_summary_csv(summary, cfg, RESULTS_DIR / f"{stem}_summary.csv")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote {RESULTS_DIR / f'{stem}_history.csv'}")
    print(f"wrote {RESULTS_DIR / f'{stem}_summary.csv'}")


if __name__ == "__main__":
    main()
