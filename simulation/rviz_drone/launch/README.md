# launch

`rviz_drone.launch.py`: starts the simulation node, Isaac Sim 6.0.1 (`isaac:=true`, default; GUI
through `../../isaac_render/isaac_python.sh` with its ROS 2 mode, the node waits for the Isaac Sim
node before it publishes) and RViz with `../rviz/rviz_drone.rviz` (`rviz:=true`, default; with
`HOME` in `../../rviz_drone_ws/ros_home`). Arguments: `trajectory` (`mission` or `figure_eights`),
`seed` (sensor-error realisation, default 0), `isaac_camera` (`chase`, `overview`, `topdown`).
Needs `source simulation/rviz_drone/env.sh` (sets `RVIZ_DRONE_REPO_ROOT`).
