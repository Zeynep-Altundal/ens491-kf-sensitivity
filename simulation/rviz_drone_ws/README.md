# rviz_drone_ws

colcon workspace of `../rviz_drone`, built with `../rviz_drone/build.sh`. Its `build/`, `install/`
and `log/` folders (including the ROS 2 logs, `log/ros`) and `ros_home/` (`ROS_HOME`, and `HOME` of
RViz so that its settings are not written to `~/.rviz2`) are generated and not version-controlled.

`fastdds_udp_localhost.xml` (version-controlled): Fast DDS profile for every ROS 2 process of this
repository (UDPv4 on 127.0.0.1 only, built-in transports off), set by `../rviz_drone/env.sh` and
`../isaac_render/isaac_python.sh` through `FASTRTPS_DEFAULT_PROFILES_FILE`.
