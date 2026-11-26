# Environment for the drone simulation with Isaac Sim 5.1.0 (standalone) and Pegasus Simulator.
#
# Source this file; it changes the current shell only and nothing global:
#     source simulation/isaac_drone/env.sh
#
# ISAACSIM_PYTHON points to isaac_python.sh, which runs the Python of Isaac Sim 5.1.0 with the
# per-user paths that come from environment variables (home, cache, data, config, temporary files)
# redirected into $ISAACSIM_PATH/_sandbox. Whether this keeps every write away from directories
# shared with other Isaac Sim installations is OPEN (see README.md).

if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "env.sh must be sourced, not executed: source ${0}" >&2
    exit 1
fi

_isaac_drone_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ISAACSIM_PATH="${ISAACSIM_PATH_OVERRIDE:-${HOME}/isaacsim-5.1.0}"
export ISAACSIM_PYTHON="${_isaac_drone_dir}/isaac_python.sh"
export PEGASUS_PATH="${PEGASUS_PATH_OVERRIDE:-${HOME}/PegasusSimulator}"

unset _isaac_drone_dir

if [ ! -x "${ISAACSIM_PATH}/python.sh" ]; then
    echo "warning: ${ISAACSIM_PATH}/python.sh not found" >&2
fi
