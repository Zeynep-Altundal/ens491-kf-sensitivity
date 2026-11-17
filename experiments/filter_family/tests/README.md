# experiments/filter_family/tests

Tests for the filter family: EKF, IEKF and UKF reproduce the KF on the linear model; the adaptive
filter with adaptation off is bitwise identical to the KF; with adaptation on it recovers R from a
100-fold overestimate; the analytic Jacobians of the range-bearing model match finite differences;
bearing wrap-around is handled in residuals and in the unscented measurement mean.
