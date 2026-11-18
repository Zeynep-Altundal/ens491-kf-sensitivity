# experiments/neural_r/tests

Tests of the differentiable Kalman filter: it reproduces the NumPy Kalman filter of
`experiments/linear_kf` with fixed R and the true-covariance filter of `experiments/filter_family`
with a time-varying R (estimates, covariances, innovations, innovation covariances and innovation
NLL to a relative tolerance of 1e-10); the learned covariance is causal (R at step t does not depend
on measurements from step t on) and positive. The tests are skipped when PyTorch is not installed.
