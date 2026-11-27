# Common interface for the filter comparison (frozen 2026-11-25, v1)

Goal: every recording (Isaac Sim, our scenario generator, EuRoC, KalmanNet data sets) is read the
same way, and every filter (KF, EKF, ESKF, KalmanNet) is called the same way and writes the same
log. Then peak analysis, NEES and plots work for all of them without special cases.

The conventions are the same as in `synthetic/README.md` and `experiments/eskf_sensitivity/README.md`
of this repository (they were first written in the earlier `ens491-nkf` repository). Code:
`experiments/nav_compare/`.

## 1. Conventions

| Item | Convention |
| --- | --- |
| World frame `w` | ENU: x East, y North, z Up. Origin at the first truth sample. |
| Body frame `b` | Frame of the IMU measurements. For the drone: the IMU frame of the simulator. |
| Orientation | Hamilton quaternion, order `[w, x, y, z]`, unit norm. `R(q)` maps `b` to `w`. |
| Gravity | `g_w = [0, 0, -9.81]` m/s² |
| Accelerometer | Specific force in `b`, gravity not removed: `f_b = R^T (a_w - g_w)` |
| Units | SI: s, m, m/s, m/s², rad, rad/s. Never degrees inside files or code. |
| Time | `t_s` float64, seconds, one common time base per recording. A constant rate is never assumed; use the measured `dt`. |
| Attitude error | `dtheta = Log(R_hat^T R_true)`, in `b` (error on the right, as in the ESKF). |
| Position / velocity error | `truth - estimate`, in `w`. |

OPEN for Zeynep: the frames that Isaac Sim / Pegasus write (ENU or NED, body FLU or FRD). They must
be converted to the table above when the recording is written, never inside a filter.

## 2. Recording file (one `.npz` per flight, readable without ROS or Isaac)

| Key | Shape | Unit | Meaning | Required |
| --- | --- | --- | --- | --- |
| `meta` | JSON string | – | See below | yes |
| `imu_t` | [N] | s | IMU time stamps | yes |
| `imu_gyro` | [N,3] | rad/s | Measured angular rate in `b` (with all errors) | yes |
| `imu_accel` | [N,3] | m/s² | Measured specific force in `b` (with all errors) | yes |
| `imu_gyro_clean`, `imu_accel_clean` | [N,3] | | Same, without any error | simulation only |
| `truth_t` | [M] | s | Truth time stamps | sim and EuRoC |
| `truth_p`, `truth_v` | [M,3] | m, m/s | True position and velocity in `w` | sim and EuRoC |
| `truth_q` | [M,4] | – | True orientation `[w, x, y, z]` | sim and EuRoC |
| `truth_bg`, `truth_ba` | [M,3] | rad/s, m/s² | True IMU biases; NaN when unknown (EuRoC) | sim |
| `pos_t` | [K] | s | Position measurement time stamps | yes |
| `pos_z` | [K,3] | m | Measured position in `w` (with all errors) | yes |
| `pos_R` | [K,3,3] | m² | Measurement covariance **told to the filter** (may differ from the true noise on purpose) | yes |
| `pos_valid` | [K] bool | – | False inside an outage; the filter must skip these | yes |
| `events` | JSON string | – | List of injected disturbances, see section 3 | yes (may be empty) |

`meta` holds: `platform` (`uav` or `ugv`), `source` (`isaac_pegasus`, `ens491_quadrotor_sim`,
`scenario_gen`, `euroc`, `kalmannet_linear`, `kalmannet_sine`, `kalmannet_lorenz`, `nclt`),
`imu_rate_hz`, `pos_rate_hz`, `seed`, `true_noise` (gyro and accel noise density, bias random walk
actually used), `imu_sample`, `version` (`v1`).

**What an IMU sample represents (`imu_sample`).** The filters hold IMU sample $k$ constant over the
interval $[t_k, t_{k+1})$ (zero-order hold). `"interval"`: sample $k$ already describes that
interval (mean rate and specific force over it), as in `scenario_gen`. `"instant"`: sample $k$ is
the value at $t_k$, as for a real IMU, EuRoC and `ens491_quadrotor_sim`; `run_filter` then uses the
mean of samples $k$ and $k+1$ for the interval. A source that averages the samples itself writes
`"interval"`. Missing field: `"interval"`. Using instantaneous samples as interval values makes the
filters overconfident in fast rotations (quadrotor simulation: full-state NEES 45.6 instead of about
15; 14.6 with the mean).

Rule: the truth (`truth_*`, `*_clean`, the true noise in `meta`) is only for evaluation. A filter
receives only `imu_*`, `pos_*` and the initial state.

## 3. Injected disturbances (`events`)

Disturbances are added to a recording afterwards, so the same flight can be replayed with different
errors. Every event is logged so that the analysis knows exactly when and where it happened.

```json
{"type": "noise_burst", "sensor": "pos", "axes": [0, 1, 2], "t_start": 30.0, "t_end": 32.0,
 "magnitude": 5.0, "unit": "m (std)"}
```

| `type` | Meaning | Applies to |
| --- | --- | --- |
| `bias_step` | Constant offset added from `t_start` to `t_end` (on and off in mid-flight) | gyro, accel |
| `noise_scale` | Noise std multiplied by `magnitude` for the whole flight | gyro, accel, pos |
| `noise_burst` | Large extra noise in a short window (the advisor's "short intervals with large errors") | gyro, accel, pos |
| `spike` | Single samples with a large wrong value; `magnitude` = size, `count` = number | gyro, accel, pos |
| `outage` | No position measurement from `t_start` to `t_end` (`pos_valid = False`) | pos |
| `scale_factor` | Measurement multiplied by `1 + magnitude` (optional this week) | gyro, accel |

## 4. Filter interface (Python, same for every filter)

```python
class Filter:
    name: str                       # "KF-CV", "KF-IMU", "EKF", "ESKF", later "KalmanNet"
    uses_imu: bool                  # False for KF-CV
    def __init__(self, x0: InitialState, P0: np.ndarray, noise: NoiseSpec): ...
    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None: ...
    def update_position(self, z: np.ndarray, R: np.ndarray) -> UpdateInfo: ...
    def state(self) -> NavState: ...       # p, v, q, b_g, b_a (NaN where not estimated)
    def cov_blocks(self) -> CovBlocks: ...  # P_pp, P_vv, P_thth, P_bg, P_ba (3x3 each, NaN if absent)
```

- `UpdateInfo` returns the innovation `innov` (symbol: y tilde), its covariance `S` and
  `NIS = innov^T S^-1 innov` (notation: `docs/notation.md`).
- The four filters of v1:

  | Name | State | Uses | Model |
  | --- | --- | --- | --- |
  | KF-CV | p, v (6) | position only | constant velocity, white acceleration noise `sigma_cv` (tuning value) |
  | KF-IMU | p, v, b_a (9) | accelerometer + position; **orientation taken from the truth** | linear because the orientation is given; shows what a linear KF can do with the IMU when the hard (rotation) part is removed |
  | EKF | p, v, q, b_g, b_a (16) | gyro + accel + position | full state, quaternion inside the state, linearised around the estimate |
  | ESKF | 15 error states | gyro + accel + position | error state, attitude error on the right (repository ESKF) |

- KF-CV ignores `gyro` and `accel` in `predict`; its `q`, biases and attitude covariance are NaN.
  KF-IMU estimates no `q` and no `b_g` (NaN).
- All filters start from the same initial position, velocity and orientation with the same
  `P_0` blocks; biases start at zero (the filter does not know them).
- Covariance blocks always use the error convention of section 1, so NEES is comparable across
  filters. For the full-state EKF the quaternion covariance is mapped to the 3-dim attitude error.

## 5. Run log (one `.npz` per filter per recording)

`t`, `p_hat`, `v_hat`, `q_hat`, `bg_hat`, `ba_hat`, `P_pp`, `P_vv`, `P_thth`, `P_bg`, `P_ba`, and
for every position update `upd_t`, `innov`, `S`, `nis`. Plus `meta` with filter name, settings and the
recording file name. The evaluator (errors, NEES, 3σ bands, peak detection, recovery time) reads
only these logs and the recording.

## 6. Decisions taken (2026-11-25)

1. File format: `.npz` (writer and reader: `experiments/nav_compare/data/recording.py`).
2. Frames: ENU, quaternion `[w, x, y, z]`; any simulator output is converted when the recording is
   written, never inside a filter.
3. Disturbances are added after recording with `experiments/nav_compare/data/disturbances.py` and logged in `events`.
4. KF: two variants, KF-CV and KF-IMU (table in section 4).
5. UGV data: from our scenario generator (`experiments/nav_compare/data/scenarios.py`).
6. Open (not blocking): Pegasus IMU noise switched off or logged, so that the true bias is known.

## 7. Notation

See `docs/notation.md` (single source) and the LaTeX macros in `docs/latex/notation.tex`.
