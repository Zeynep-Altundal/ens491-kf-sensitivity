#!/usr/bin/env bash
# Runs the Python interpreter of Isaac Sim 5.1.0 in an isolated per-user environment.
#
# The per-user locations that are read from environment variables are redirected for this process
# only, into $ISAACSIM_PATH/_sandbox: HOME, XDG cache/data/config, TMPDIR, the NVIDIA driver shader
# cache and the pip cache. SimulationApp additionally starts Kit in portable mode.
#
# OPEN: this is not sufficient on its own. The Kit application plugin resolves the home directory
# with getpwuid_r and not with $HOME, and the telemetry transmitter writes to ${omni_logs}; whether
# these stay inside the installation directory has not been verified (see README.md).
#
# ROS 2 variables inherited from the calling shell are removed and a private ROS domain on the
# loopback interface is set, so that the simulator never joins the ROS 2 network of other work.
#
# Usage (after `source simulation/isaac_drone/env.sh`):
#     "$ISAACSIM_PYTHON" script.py [args]
#     "$ISAACSIM_PYTHON" -m pip install --editable <path>
set -euo pipefail

: "${ISAACSIM_PATH:?ISAACSIM_PATH is not set; source simulation/isaac_drone/env.sh first}"
if [ ! -x "${ISAACSIM_PATH}/python.sh" ]; then
    echo "error: ${ISAACSIM_PATH}/python.sh not found" >&2
    exit 1
fi

SANDBOX="${ISAACSIM_PATH}/_sandbox"
mkdir -p "${SANDBOX}/home" "${SANDBOX}/cache/nvidia" "${SANDBOX}/cache/pip" \
         "${SANDBOX}/data" "${SANDBOX}/config" "${SANDBOX}/tmp"

export HOME="${SANDBOX}/home"
export XDG_CACHE_HOME="${SANDBOX}/cache"
export XDG_DATA_HOME="${SANDBOX}/data"
export XDG_CONFIG_HOME="${SANDBOX}/config"
export TMPDIR="${SANDBOX}/tmp"
export __GL_SHADER_DISK_CACHE_PATH="${SANDBOX}/cache/nvidia"
export PIP_CACHE_DIR="${SANDBOX}/cache/pip"
export PYTHONNOUSERSITE=1

# Remove the ROS 2 environment of the calling shell.
for v in $(compgen -e); do
    case "$v" in
        ROS_*|AMENT_*|COLCON_*|RMW_*|CYCLONEDDS_*|FASTRTPS_*|FASTDDS_*) unset "$v" ;;
    esac
done
unset PYTHONPATH LD_LIBRARY_PATH CMAKE_PREFIX_PATH
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '^/opt/ros/' | paste -sd: -)"
export PATH
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=101

exec "${ISAACSIM_PATH}/python.sh" "$@"
