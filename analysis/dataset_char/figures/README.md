# analysis/dataset_char/figures

PNG figures written by `characterise.py`:

| File | Content |
|------|---------|
| `<dataset>_timing.png` | Histogram of the IMU timestamp jitter and the sampling interval over time (`kitti`, `euroc`). |
| `<dataset>_histograms.png` | Histograms of the gyroscope and accelerometer signals per axis with the excess kurtosis. |
| `<dataset>_motion.png` | Ground-truth speed, gyroscope rate norm with the turning threshold, and finite-difference acceleration norm over time. |
| `motion_comparison.png` | Ranges of speed, angular rate and acceleration and the fraction of time turning for KITTI, EuRoC and the synthetic profiles. |

No Allan deviation figure is written: neither sequence contains a static segment of at least 10 s.
