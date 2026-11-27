# Environment for rendering the quadrotor simulation with the pip installation of Isaac Sim 6.0.1.
#
# Source this file; it changes the current shell only:
#     source simulation/isaac_render/env.sh
#
# ISAACSIM_PYTHON points to isaac_python.sh, which runs the venv Python of the installation in
# ~/isaacsim-6.0.1-ens491 with every per-user location redirected into its _sandbox.

if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "env.sh must be sourced, not executed: source ${0}" >&2
    exit 1
fi

_isaac_render_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ISAACSIM601_ROOT="${ISAACSIM601_ROOT_OVERRIDE:-${HOME}/isaacsim-6.0.1-ens491}"
export ISAACSIM_PYTHON="${_isaac_render_dir}/isaac_python.sh"
export RVIZ_DRONE_REPO_ROOT="$(cd "${_isaac_render_dir}/../.." && pwd)"

unset _isaac_render_dir

if [ ! -x "${ISAACSIM601_ROOT}/venv/bin/python" ]; then
    echo "warning: ${ISAACSIM601_ROOT}/venv/bin/python not found" >&2
fi
