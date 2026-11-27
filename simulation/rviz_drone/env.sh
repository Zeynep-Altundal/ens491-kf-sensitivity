# Environment for the rviz_drone package. Source it; it changes the current shell only:
#     source simulation/rviz_drone/env.sh
#
# Sources ROS 2 Jazzy and, if it has been built, the overlay of simulation/rviz_drone_ws, and puts
# all ROS 2 traffic of this shell on a private domain restricted to this machine, so that it never
# reaches nodes of other work running on the default domain.

if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "env.sh must be sourced, not executed: source ${0}" >&2
    exit 1
fi

_rviz_drone_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RVIZ_DRONE_WS="$(cd "${_rviz_drone_dir}/../rviz_drone_ws" && pwd)"
export RVIZ_DRONE_REPO_ROOT="$(cd "${_rviz_drone_dir}/../.." && pwd)"

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
if [ -f "${RVIZ_DRONE_WS}/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${RVIZ_DRONE_WS}/install/setup.bash"
fi

export ROS_DOMAIN_ID=101
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# Keep the ROS 2 logs and the ROS home (ros2 daemon, etc.) of this package in its workspace instead
# of the shared ~/.ros.
export ROS_LOG_DIR="${RVIZ_DRONE_WS}/log/ros"
export ROS_HOME="${RVIZ_DRONE_WS}/ros_home"
# Fast DDS (also bundled with Isaac Sim 6.0.1, same version 2.14.6). UDP only is requested, but with
# ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST shared-memory port files in /dev/shm were still rewritten by
# the ros2 command-line tools (README.md, "Isolation"); /dev/shm is also used by other ROS 2 work.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

unset _rviz_drone_dir
