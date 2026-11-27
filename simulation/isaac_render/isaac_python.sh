#!/usr/bin/env bash
# Runs the Python of the pip installation of Isaac Sim 6.0.1 (venv) in an isolated per-user environment.
# Same isolation settings as ../isaac_drone/isaac_python.sh (verified there with 0 differences).
#
# The per-user locations that are read from environment variables are redirected for this process
# only, into $ISAACSIM601_ROOT/_sandbox: HOME, XDG cache/data/config, TMPDIR, the NVIDIA driver shader
# cache and the pip cache. For a script run, Kit settings (portable root, structured log, telemetry,
# path tokens) move the remaining per-user locations into the sandbox as well. For Isaac Sim 5.1.0
# these settings gave 0 differences in the shared directories; for 6.0.1 see README.md.
#
# ROS 2 variables inherited from the calling shell are removed and a private ROS domain on the
# loopback interface is set, so that the simulator never joins the ROS 2 network of other work.
#
# Usage (after `source simulation/isaac_render/env.sh`):
#     "$ISAACSIM_PYTHON" script.py [args]
#     "$ISAACSIM_PYTHON" -m pip install --editable <path>
set -euo pipefail

: "${ISAACSIM601_ROOT:?ISAACSIM601_ROOT is not set; source simulation/isaac_render/env.sh first}"
PYTHON_EXE="${ISAACSIM601_ROOT}/venv/bin/python"
if [ ! -x "${PYTHON_EXE}" ]; then
    echo "error: ${PYTHON_EXE} not found" >&2
    exit 1
fi

SANDBOX="${ISAACSIM601_ROOT}/_sandbox"
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
export CUDA_CACHE_PATH="${SANDBOX}/cache/ComputeCache"

# Kit settings that move every remaining per-user location into the sandbox. They are appended
# only when a script is run (SimulationApp passes unknown arguments on to Kit), not for `-m pip`.
PORTABLE="${SANDBOX}/portable"
GLOBAL="${SANDBOX}/global"
mkdir -p "${PORTABLE}/logs" "${PORTABLE}/structuredlog" "${GLOBAL}"/{config,data,logs,cache,documents}
PRIVACY="${SANDBOX}/config/privacy.toml"
if [ ! -f "${PRIVACY}" ]; then
    printf '[privacy]\nperformance = false\npersonalization = false\nusage = false\n' > "${PRIVACY}"
fi
KIT_ISOLATION_ARGS=(
    "--portable-root" "${PORTABLE}"
    "--/structuredLog/enable=false"
    "--/structuredLog/logDirectory=${PORTABLE}/structuredlog"
    "--/structuredLog/privacySettingsFile=${PRIVACY}"
    "--/telemetry/enableAnonymousData=false"
    "--/telemetry/log/file=${PORTABLE}/logs/omni.telemetry.transmitter.log"
    "--/app/tokens/omni_global_config=${GLOBAL}/config"
    "--/app/tokens/omni_global_data=${GLOBAL}/data"
    "--/app/tokens/omni_global_logs=${GLOBAL}/logs"
    "--/app/tokens/omni_global_cache=${GLOBAL}/cache"
    "--/app/tokens/omni_config=${PORTABLE}/config"
    "--/app/tokens/omni_data=${PORTABLE}/data"
    "--/app/tokens/omni_logs=${PORTABLE}/logs"
    "--/app/tokens/omni_cache=${PORTABLE}/cache"
    "--/app/tokens/omni_documents=${GLOBAL}/documents"
    "--/app/tokens/shared_documents=${GLOBAL}/documents/shared"
    "--/app/tokens/app_documents=${GLOBAL}/documents/app"
    "--/app/tokens/documents=${GLOBAL}/documents"
)

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

# ROS 2 mode (ISAACSIM_ROS2=1, set by the launch file of ../rviz_drone): the ROS 2 bridge of Isaac
# Sim is meant to use the Jazzy libraries bundled with its extension isaacsim.ros2.core (Fast DDS
# 2.14.6, the same version as the system Jazzy), never /opt/ros (not yet verified at run time, see
# README.md). Same domain, discovery range, Fast DDS profile and log location as ../rviz_drone/env.sh.
if [ "${ISAACSIM_ROS2:-0}" = "1" ]; then
    _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export ROS_DISTRO=jazzy
    export LD_LIBRARY_PATH="${ISAACSIM601_ROOT}/venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export FASTRTPS_DEFAULT_PROFILES_FILE="${_repo}/simulation/rviz_drone_ws/fastdds_udp_localhost.xml"
    unset ROS_LOCALHOST_ONLY
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    export ROS_LOG_DIR="${_repo}/simulation/rviz_drone_ws/log/ros"
    export ROS_HOME="${_repo}/simulation/rviz_drone_ws/ros_home"
    unset _repo
fi

if [[ "${1:-}" == *.py ]]; then
    exec "${PYTHON_EXE}" "$@" "${KIT_ISOLATION_ARGS[@]}"
fi
exec "${PYTHON_EXE}" "$@"
