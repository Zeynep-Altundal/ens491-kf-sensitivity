# rviz

`rviz_drone.rviz`: RViz configuration (fixed frame `world`; grid of 5 m cells over 70 m x 70 m
centred on the mission area, TF, drone marker, truth path, position measurements, and the paths
of the reserved estimator topics `/drone/estimate/<kf|ekf|eskf|spare>/path` in their reserved
colours, see `../rviz_drone/topics.py`). Orbit view centred at (20, 10, 1.5) m, 62 m away, from
the south, so that the whole mission (about 42 m x 28 m) is in view.
