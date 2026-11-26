#!/usr/bin/env bash
# Build rviz_drone in its own workspace simulation/rviz_drone_ws (build/, install/, log/ there).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ws="$(cd "${here}/../rviz_drone_ws" && pwd)"

# shellcheck disable=SC1091
set +u
source /opt/ros/jazzy/setup.bash
set -u

cd "${ws}"
colcon build --symlink-install --base-paths "${here}" \
    --build-base "${ws}/build" --install-base "${ws}/install" "$@"
