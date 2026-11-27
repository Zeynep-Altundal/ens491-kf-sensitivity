# experiments/estimators/tests

Unit tests of the estimators: quaternion maps, EKF Jacobian against central finite differences
(tangent directions of the quaternion), the interface with missing position measurements for the
own filters, and the own KF against `experiments/linear_kf/kf.py`. The companion filters are
checked by `../sanity.py` on the recordings (they need the pinned export in `../_vendor/`).
