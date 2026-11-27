# Notation (single source for code and report)

Every symbol has exactly one meaning, in every filter (KF, EKF, ESKF) and in every section of the
report. The LaTeX macros in `docs/latex/notation.tex` produce these symbols; please use the
macros in Overleaf instead of typing the symbols by hand. Conventions for frames, quaternions and
units are in `docs/interfaces.md`, section 1.

## Time, frames, rotation

| Symbol | Meaning | Unit | Code name |
| --- | --- | --- | --- |
| $k$ | Time step index | – | `k` |
| $t_k$, $\Delta t_k = t_{k+1}-t_k$ | Time stamp, measured step length (never assumed constant) | s | `t`, `dt` |
| $w$ | World frame, ENU (x East, y North, z Up) | – | `w` |
| $b$ | Body frame (frame of the IMU) | – | `b` |
| $\mathbf{R}\in SO(3)$ | Orientation, rotation from $b$ to $w$ | – | `R` |
| $\mathbf{q}$ | Same orientation as a unit quaternion, scalar first $[q_0,q_1,q_2,q_3]$ | – | `q` |
| $[\mathbf{a}]_\times$ | Skew-symmetric matrix, $[\mathbf{a}]_\times\mathbf{b}=\mathbf{a}\times\mathbf{b}$ | – | `skew` |
| $\operatorname{Exp}(\boldsymbol{\phi})$, $\operatorname{Log}(\mathbf{R})$ | Rotation vector to rotation matrix and back | – | `exp_so3`, `log_so3` |
| $\mathbf{g}$ | Gravity in $w$, $[0,0,-9.81]^\top$ | m/s² | `GRAVITY_W` |

## States and estimates

| Symbol | Meaning | Unit | Code name |
| --- | --- | --- | --- |
| $\mathbf{x}$ | State (the quantities the filter estimates) | mixed | `x` |
| $\mathbf{p}$, $\mathbf{v}$ | Position and velocity in $w$ | m, m/s | `p`, `v` |
| $\mathbf{b}_g$, $\mathbf{b}_a$ | Gyroscope and accelerometer bias in $b$ | rad/s, m/s² | `b_g`, `b_a` |
| $\hat{\mathbf{x}}_k^-$ | Prediction: estimate at $t_k$ before the measurement | – | `x_pred` |
| $\hat{\mathbf{x}}_k^+$ | Update: estimate at $t_k$ after the measurement | – | `x_upd` |
| $\mathbf{P}_k^-$, $\mathbf{P}_k^+$ | Covariance of the estimation error, before and after the update | – | `P` |
| $\hat{\mathbf{x}}_0$, $\mathbf{P}_0$ | Initial estimate and its covariance | – | `x0`, `P0` |
| $\delta\mathbf{x}$ | Error state of the ESKF, $[\delta\mathbf{p},\delta\mathbf{v},\delta\boldsymbol{\theta},\delta\mathbf{b}_g,\delta\mathbf{b}_a]$ (15) | mixed | `dx` |
| $\delta\boldsymbol{\theta}$ | Attitude error, $\mathbf{R}=\hat{\mathbf{R}}\operatorname{Exp}(\delta\boldsymbol{\theta})$ (error on the right) | rad | `dtheta` |
| $\mathbf{e}_k$ | Estimation error used for evaluation: true minus estimate (attitude: $\operatorname{Log}(\hat{\mathbf{R}}^\top\mathbf{R})$) | mixed | `err` |

## Models and noise

| Symbol | Meaning | Unit | Code name |
| --- | --- | --- | --- |
| $\mathbf{f}(\cdot)$ | Motion model (how the state moves from $t_k$ to $t_{k+1}$) | – | `f` |
| $\mathbf{F}_k$ | Discrete state transition matrix (KF), or Jacobian $\partial\mathbf{f}/\partial\mathbf{x}$ (EKF, ESKF) | – | `F` |
| $\mathbf{A}$ | Continuous-time error-state matrix of the ESKF, $\mathbf{F}_k\approx\mathbf{I}+\mathbf{A}\,\Delta t_k$ | – | `A` |
| $\mathbf{G}$ | Matrix that maps the noise into the state | – | `G` |
| $\mathbf{u}_k$ | Known input of the KF with IMU (acceleration in $w$) | m/s² | `u` |
| $\mathbf{Q}_c$, $\mathbf{Q}_k$ | Process noise covariance, continuous and discrete ($\mathbf{Q}_k\approx\mathbf{G}\mathbf{Q}_c\mathbf{G}^\top\Delta t_k$) | – | `Q_c`, `Q` |
| $\boldsymbol{\omega}_m$, $\mathbf{f}_m$ | Measured angular rate and specific force (IMU output, in $b$) | rad/s, m/s² | `gyro`, `accel` |
| $\mathbf{n}_g$, $\mathbf{n}_a$ | White noise of gyroscope and accelerometer | – | – |
| $\mathbf{n}_{bg}$, $\mathbf{n}_{ba}$ | Noise that drives the bias random walk | – | – |
| $\sigma_g$, $\sigma_a$ | Noise density of gyroscope and accelerometer | rad/s/√Hz, m/s²/√Hz | `gyro_density`, `accel_density` |
| $\sigma_{bg}$, $\sigma_{ba}$ | Bias random walk density | rad/s²/√Hz, m/s³/√Hz | `gyro_bias_rw`, `accel_bias_rw` |
| $\sigma_{cv}$ | Acceleration noise of the constant-velocity KF (a tuning value) | m/s²/√Hz | `sigma_cv` |

## Measurements and update

| Symbol | Meaning | Unit | Code name |
| --- | --- | --- | --- |
| $\mathbf{z}_k$ | Measurement (here: position in $w$) | m | `z` |
| $\mathbf{h}(\cdot)$ | Measurement model (what the measurement should be for a given state) | – | `h` |
| $\mathbf{H}_k$ | Measurement matrix, or Jacobian $\partial\mathbf{h}/\partial\mathbf{x}$ | – | `H` |
| $\boldsymbol{\eta}_k$ | Measurement noise, $\boldsymbol{\eta}_k\sim\mathcal{N}(\mathbf{0},\mathbf{R}_k)$ | m | – |
| $\mathbf{R}_k$ | Measurement noise covariance **told to the filter** | m² | `R_meas` |
| $\tilde{\mathbf{y}}_k$ | Innovation, $\mathbf{z}_k-\mathbf{h}(\hat{\mathbf{x}}_k^-)$ | m | `innov` |
| $\mathbf{S}_k$ | Innovation covariance, $\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^\top+\mathbf{R}_k$ | m² | `S` |
| $\mathbf{K}_k$ | Kalman gain, $\mathbf{P}_k^-\mathbf{H}_k^\top\mathbf{S}_k^{-1}$ | – | `K` |

Note: $\mathbf{R}$ (orientation) and $\mathbf{R}_k$ (measurement covariance) look alike. The
measurement covariance always carries the time index $k$; the orientation never does.

## Evaluation

| Symbol | Meaning | Expected value |
| --- | --- | --- |
| RMSE | Root mean square of the position error over a run | – |
| NEES | $\mathbf{e}_k^\top\mathbf{P}_k^{-1}\mathbf{e}_k$, per block (position, velocity, attitude) | 3 for one 3-D block |
| NIS | $\tilde{\mathbf{y}}_k^\top\mathbf{S}_k^{-1}\tilde{\mathbf{y}}_k$ | 3 for a 3-D position measurement |
| $3\sigma$ bound | $\pm3\sqrt{\mathbf{P}_{ii}}$; the error should stay inside it about 99 % of the time | – |
| Peak | A local maximum of the position error that is clearly larger than the error around it | – |
| Recovery time | Time after a disturbance ends until the error is back near its level before the disturbance | s |

Symbols not used, on purpose: $\mathbf{v}$ is only velocity (never measurement noise), $\mathbf{w}$
is only the world frame (never process noise), $\nu$ is not used (looks like $v$), $\epsilon$ is
not used.
