# Notation (report and code)

One notation for the whole project: the one of the progress report, which follows the KalmanNet
paper (Revach et al., IEEE TSP 2022). The LaTeX macros are in `docs/latex/notation.tex` (copied from
the report's `main.tex`). The column "Code" gives the name used in the code (`experiments/`),
where some names differ for practical reasons (for example `P` for the covariance).
Frames, quaternion convention and units: `docs/common_interface.md`, section 1.

## General filter

| Symbol | Meaning | Code |
| --- | --- | --- |
| $t$, $T$ | time step, number of steps | `k`, `t` (time stamp in s) |
| $\mathbf{x}_t\in\mathbb{R}^m$, $\mathbf{y}_t\in\mathbb{R}^n$ | hidden state, observation (measurement) | `x`, `z` / `y` |
| $\mathbf{f}$, $\mathbf{h}$ | state-evolution and observation maps | `f`, `h` |
| $\mathbf{F}$, $\mathbf{H}$; $\hat{\mathbf{F}}_t$, $\hat{\mathbf{H}}_t$ | matrices of the linear model; Jacobians at the estimate | `F`, `H` |
| $\mathbf{w}_t\sim\mathcal{N}(\mathbf{0},\mathbf{Q})$ | process noise | `Q` |
| $\mathbf{v}_t\sim\mathcal{N}(\mathbf{0},\mathbf{R})$ | observation noise (in the navigation sections $\mathbf{v}$ is the velocity, see below) | `R_meas`, `pos_R` |
| $\hat{\mathbf{x}}_{t\mid t-1}$, $\hat{\mathbf{x}}_{t\mid t}$ | prior (predicted) and posterior (filtered) estimate | `x_pred`, `x_upd`, `p_hat`... |
| $\boldsymbol{\Sigma}_{t\mid t-1}$, $\boldsymbol{\Sigma}_{t\mid t}$ | error covariance of the prior and of the posterior | `P` |
| $\hat{\mathbf{x}}_0$, $\boldsymbol{\Sigma}_0$ | initial estimate and covariance | `x0`, `P0` |
| $\Delta\mathbf{y}_t=\mathbf{y}_t-\mathbf{h}(\hat{\mathbf{x}}_{t\mid t-1})$ | innovation | `innov` |
| $\mathbf{S}_t$, $\mathbf{K}_t$ | innovation covariance, gain | `S`, `K` |
| $\mathcal{K}_t(\boldsymbol{\Theta})$ | gain of KalmanNet with weights $\boldsymbol{\Theta}$ | – |

## Inertial navigation (ESKF and the drone / car experiments)

| Symbol | Meaning | Unit | Code |
| --- | --- | --- | --- |
| $\mathbf{p}$, $\mathbf{v}$ | position and velocity in the world frame (ENU, z up) | m, m/s | `p`, `v` |
| $\mathbf{q}$, $\mathbf{C}=\mathbf{C}(\mathbf{q})$ | attitude: unit quaternion (Hamilton, scalar first) and rotation matrix, body to world | – | `q`, `R` |
| $\mathbf{b}_g$, $\mathbf{b}_a$ | gyroscope and accelerometer bias (body frame) | rad/s, m/s² | `b_g`, `b_a` |
| $\boldsymbol{\omega}_m$, $\mathbf{a}_m$ | measured angular rate and specific force (IMU output) | rad/s, m/s² | `gyro`, `accel` |
| $\boldsymbol{\omega}_n$, $\mathbf{a}_n$ | white noise of gyroscope and accelerometer | – | densities `gyro_density`, `accel_density` |
| $\boldsymbol{\omega}_w$, $\mathbf{a}_w$ | noise driving the bias random walks | – | `gyro_bias_rw`, `accel_bias_rw` |
| $\mathbf{g}$ | gravity, $[0,0,-9.81]^\top$ | m/s² | `GRAVITY_W` |
| $\delta\mathbf{x}=(\delta\mathbf{p},\delta\mathbf{v},\delta\boldsymbol{\theta},\delta\mathbf{b}_g,\delta\mathbf{b}_a)$ | error state of the ESKF (15) | mixed | `dx` |
| $\delta\boldsymbol{\theta}$ | attitude error, $\mathbf{q}_\text{true}=\mathbf{q}\otimes\operatorname{Exp}(\delta\boldsymbol{\theta})$ (body frame) | rad | `dtheta` |
| $\mathbf{A}$ | continuous error-state matrix, $\mathbf{F}_t\approx\mathbf{I}+\mathbf{A}\Delta t$ | – | `A` |
| $[\cdot]_\times$, $\operatorname{Exp}$, $\operatorname{Log}$ | skew-symmetric matrix; rotation vector to rotation and back | – | `skew`, `exp`, `log` |

In the code `R` is the rotation matrix and `R_meas` / `pos_R` the measurement covariance; in the
report the rotation matrix is $\mathbf{C}$ because $\mathbf{R}$ is the noise covariance.

## Evaluation

| Symbol | Meaning | Expected value |
| --- | --- | --- |
| $\mathbf{e}_t=\mathbf{x}_t-\hat{\mathbf{x}}_{t\mid t}$ | error, truth minus estimate (attitude: $\operatorname{Log}(\hat{\mathbf{C}}^\top\mathbf{C})$) | – |
| NEES | $\mathbf{e}_t^\top\boldsymbol{\Sigma}_{t\mid t}^{-1}\mathbf{e}_t$, per block (position, velocity, attitude) | 3 for a 3-D block |
| NIS | $\Delta\mathbf{y}_t^\top\mathbf{S}_t^{-1}\Delta\mathbf{y}_t$ | the measurement dimension |
| MSE [dB], RMSE | $10\log_{10}$ of the mean squared error; root mean square error in physical units | – |
| Peak | maximal run of steps with $e_t>\max(\mathrm{median}+5\cdot1.4826\,\mathrm{MAD},\ 2\,\mathrm{median})$ (`experiments/nav_compare/peaks.py`) | – |
| Recovery time | time after a disturbance until the error is back near its level before it | s |
