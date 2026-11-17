# Neural Kalman Filtering for Inertial Navigation in GNSS-Denied Environments - Classical vs Neural ESKF Study

This repository contains the mathematical background, dataset characterisation, synthetic
scenario generation and sensitivity experiments for a comparative study of classical and
neural error-state Kalman filters (ESKF) for inertial navigation without GNSS.

## Repository layout

| Path | Purpose |
|------|---------|
| `docs/` | Project documentation. |
| `docs/math/` | KF, EKF, ESKF and neural KF equations, and the rationale behind component choices. |
| `analysis/` | Analyses of real-world datasets. |
| `analysis/dataset_char/` | IMU characterisation of the KITTI and EuRoC datasets. |
| `synthetic/` | Scenario generator: noise, bias drift, outliers, dropout and motion profiles. |
| `experiments/` | Experiment code and results. |
| `experiments/eskf_sensitivity/` | One-parameter-at-a-time sensitivity sweeps (Q, R, P0, bias, dropout, gating) and plots. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
