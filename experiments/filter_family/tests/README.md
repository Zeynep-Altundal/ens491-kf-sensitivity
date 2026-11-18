# experiments/filter_family/tests

Tests for the filter family: EKF, IEKF and UKF reproduce the KF on the linear model; the adaptive
filter with adaptation off is bitwise identical to the KF; with adaptation on it recovers R from a
100-fold overestimate; the analytic Jacobians of the range-bearing model match finite differences;
bearing wrap-around is handled in residuals and in the unscented measurement mean.

`test_time_varying_r.py`: the time-varying simulator with unit scale reproduces `simulate`; the step,
ramp and burst patterns match their definitions; the true-covariance filter with constant R equals the EKF; the innovation NLL equals
the Gaussian negative log-density of the innovation under the filter innovation covariance S; the
recorded R equals the adaptive filter's history and the windowed estimator used in the diagnosis;
the adaptive estimate has no negative eigenvalue.
