# Simulation

Simulation environments that produce recordings (truth, IMU and other sensors) for the filters of
this repository.

| Path | Purpose |
|------|---------|
| `isaac_drone/` | Multirotor simulation with NVIDIA Isaac Sim 5.1.0 (standalone) and Pegasus Simulator v5.1.0; abandoned after a start-up crash (see its README). |
| `rviz_drone/` | ROS 2 Jazzy package: 6-DoF quadrotor with cascaded controller, IMU and position measurements from `synthetic/`, RViz visualisation. |
| `rviz_drone_ws/` | colcon workspace in which `rviz_drone/` is built (generated folders are not version-controlled). |
