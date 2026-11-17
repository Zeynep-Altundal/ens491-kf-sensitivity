# experiments/eskf_sensitivity

Classical error-state Kalman filter (ESKF) for IMU propagation with low-rate position updates,
validated on the synthetic generator in `synthetic/`. The folder will also hold the
one-parameter-at-a-time sensitivity sweeps (Q, R, P0, bias, dropout, gating); these are not yet
implemented.

## Conventions

As in `synthetic/README.md`: ENU world frame, gravity `g = [0, 0, -9.81] m/s^2`, Hamilton unit
quaternion `q` (scalar first) for the body-to-world rotation `R = R(q)`, gyroscope measuring the
body rate `w_m = w_b + b_g + n_g`, accelerometer measuring the specific force
`f_m = R^T (a - g) + b_a + n_a`.

| Symbol | Meaning |
|--------|---------|
| `x = (p, v, q, b_g, b_a)` | nominal state (16 numbers) |
| `dx = [dp, dv, dtheta, db_g, db_a]` | error state (15) |
| `[u]x` | skew-symmetric matrix, `[u]x w = u x w` |
| `Exp(u)` / `Log(q)` | rotation vector to quaternion and back |

Error definition (local attitude error, expressed in the body frame):

    p_true = p + dp,   v_true = v + dv,   q_true = q ⊗ Exp(dtheta),
    b_g,true = b_g + db_g,   b_a,true = b_a + db_a.

## Propagation (`propagate`)

The filter stores the previous IMU sample `(w_0, f_0)`; a call with the new sample `(w_1, f_1)`
propagates over `dt`.

Nominal state (same second-order scheme as the generator tests):

    w_i  = w_m,i - b_g,                     i = 0, 1
    phi  = dt/2 (w_0 + w_1) + dt^2/12 (w_0 x w_1)
    q_1  = q_0 ⊗ Exp(phi)
    a_0  = R(q_0) (f_m,0 - b_a) + g,        a_1 = R(q_1) (f_m,1 - b_a) + g
    p_1  = p_0 + v_0 dt + dt^2 (a_0/3 + a_1/6)
    v_1  = v_0 + dt/2 (a_0 + a_1)
    b_g, b_a constant

Continuous error dynamics, with `w = (w_0 + w_1)/2`, `f = (f_m,0 + f_m,1)/2 - b_a` and
`R = R(q_0)`:

    d(dp)/dt     = dv
    d(dv)/dt     = -R [f]x dtheta - R db_a - R n_a
    d(dtheta)/dt = -[w]x dtheta - db_g - n_g
    d(db_g)/dt   = n_bg
    d(db_a)/dt   = n_ba

i.e. `d(dx)/dt = A dx + noise` with the non-zero blocks of `A`:
`A_p,v = I`, `A_v,theta = -R [f]x`, `A_v,ba = -R`, `A_theta,theta = -[w]x`, `A_theta,bg = -I`.

Discrete transition and covariance propagation:

    F = I + A dt + (A dt)^2 / 2
    P <- F P F^T + Q
    Q = diag(0, s_a^2 dt I, s_g^2 dt I, s_bg^2 dt I, s_ba^2 dt I)

`s_g`, `s_a` are the gyro and accelerometer noise densities and `s_bg`, `s_ba` the bias
random-walk densities, in the units of `synthetic/README.md`. The velocity block uses
`R (s_a^2 I) R^T = s_a^2 I`. The second-order term in `F` provides the `dt^2` couplings
(for example `dp/dtheta = -R [f]x dt^2/2`) that the nominal position step contains.

## Position update (`update_position`)

    z = p_true + v,   v ~ N(0, R_z),   H = [I 0 0 0 0]
    nu  = z - p                          (innovation)
    S   = H P H^T + R_z,   NIS = nu^T S^-1 nu
    K   = P H^T S^-1
    dx  = K nu
    P   <- (I - K H) P (I - K H)^T + K R_z K^T          (Joseph form)

Error injection: `p += dp`, `v += dv`, `q <- q ⊗ Exp(dtheta)`, `b_g += db_g`, `b_a += db_a`.
Reset: the error mean is set to zero and `P <- G P G^T` with
`G = blkdiag(I_6, I - [dtheta/2]x, I_6)`.

Optional innovation gate (`gate_prob`, off by default): the update is skipped when
`NIS > chi2.ppf(gate_prob, 3)`.

## Initialisation used in the runs (`run.py`)

`P0 = diag(s_p0^2 I, s_v0^2 I, s_th0^2 I, s_bg0^2 I, s_ba0^2 I)` with
`s_p0 = 0.5 m`, `s_v0 = 0.1 m/s`, `s_th0 = 0.01 rad`, `s_bg0 = 2e-3 rad/s`, `s_ba0 = 0.05 m/s^2`.
For run index `i` the scenario uses seed `i`; the turn-on biases and the initial position,
velocity and attitude errors are drawn from these `P0` blocks, so the filter's initial
uncertainty matches the actual initial error. The bias estimates start at zero. The filter is
given the true IMU noise parameters and the true measurement covariance. Metrics are evaluated at
the measurement epochs (10 Hz) after the update.

## Files

| File | Purpose |
|------|---------|
| `eskf.py` | Filter: nominal step, error dynamics, process noise, injection, reset, update, gate. |
| `run.py` | Scenario set-up, filter run, summary metrics and CSV output. |
| `plots.py` | Figures of a single run. |
| `tests/` | Validation tests. |
| `results/` | CSV output. |
| `figures/` | PNG output. |

## Running (from the repository root)

```bash
python -m pytest -v -s experiments/eskf_sensitivity
python -m experiments.eskf_sensitivity.run
python -m experiments.eskf_sensitivity.plots
```
