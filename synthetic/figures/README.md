# synthetic/figures

PNG figures written by `plots.py`, one set per profile (`straight`, `turning`, `aggressive`), for a
60 s scenario with all error sources enabled, IMU outlier probability 1e-3, measurement outlier
probability 0.02 and a measurement dropout window from 20 s to 30 s:

| File | Content |
|------|---------|
| `<profile>_trajectory.png` | True trajectory and position measurements (top view and versus time). |
| `<profile>_gyro.png` | Clean versus corrupted gyroscope signals, and their difference with the true bias. |
| `<profile>_accel.png` | Same for the accelerometer. |
| `<profile>_bias.png` | True gyroscope and accelerometer bias over time. |
