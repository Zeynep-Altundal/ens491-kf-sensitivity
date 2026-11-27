# experiments/estimators/tests

Unit tests of the estimators: quaternion maps, EKF Jacobian against central finite differences
(tangent directions of the quaternion), the interface with missing position measurements for the
own filters, the own KF against `experiments/linear_kf/kf.py`, and the interval-mean IMU input of
the companion adapter (with a stand-in filter object). The companion filters themselves are
checked by `../sanity.py` on the recordings (they need the pinned export in `../_vendor/`).
