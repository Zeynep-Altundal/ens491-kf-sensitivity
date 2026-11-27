#!/usr/bin/env bash
# Downloads the Isaac Sim 6.0 assets used by render_flight.py into $ISAACSIM601_ROOT/assets
# (outside the repository; the assets are not redistributed here) and checks their sha256.
#
# Usage (after `source simulation/isaac_render/env.sh`):
#     bash simulation/isaac_render/fetch_assets.sh
set -euo pipefail

: "${ISAACSIM601_ROOT:?ISAACSIM601_ROOT is not set; source simulation/isaac_render/env.sh first}"
DEST="${ISAACSIM601_ROOT}/assets"
BASE="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"

# local path | remote path (below BASE) | sha256
FILES=(
"ARL-Robot-1/arl_robot_1.usd|Isaac/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd|8de2ffbc26a2bdbd82d741678f7aa9e47ea58a200170c9f9c776e8addd5b9a75"
"ARL-Robot-1/configuration/arl_robot_1_base.usd|Isaac/Robots/NTNU/ARL-Robot-1/configuration/arl_robot_1_base.usd|9a9315b51fbe02a013e29af382966e2fbc36b8304cf25d728dacbe8818eb6533"
"ARL-Robot-1/configuration/arl_robot_1_physics.usd|Isaac/Robots/NTNU/ARL-Robot-1/configuration/arl_robot_1_physics.usd|1ecab07a4a7ccf526f3cc2fd0a0aed3938cd01739bfcbf0082826c56cefbfd47"
"ARL-Robot-1/configuration/arl_robot_1_robot.usd|Isaac/Robots/NTNU/ARL-Robot-1/configuration/arl_robot_1_robot.usd|bda6c392b3fa9e886caf97c3e8cc0c8ff0181ba35971380359c7a7d28a645512"
"ARL-Robot-1/configuration/arl_robot_1_sensor.usd|Isaac/Robots/NTNU/ARL-Robot-1/configuration/arl_robot_1_sensor.usd|452de32b4e52d79a407ee36e1820f32775baed844dedc7497378a9909664425c"
"ARL-Robot-1/LICENSE|Isaac/Robots/NTNU/ARL-Robot-1/LICENSE|297161f5a4f59673fbc2bb245a20023dd3f4ce156f0249fa3431b9a25c70174e"
"Terrains/flat_plane.usd|Isaac/Environments/Terrains/flat_plane.usd|d2d14a058e217296ae279a5d7f031c26c7db3ce80c893096d74c593aa3e52a73"
"Terrains/rough_plane.usd|Isaac/Environments/Terrains/rough_plane.usd|c39ad8c2c3c262871df515e73b6fc686e796c94bac0123ed06e938d348f80126"
"Skies/kloofendal_48d_partly_cloudy_4k.hdr|NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr|c656b652aba1915cd0c27faad7be256b6f2796eb711cef6a2aadc18dd80a4187"
"Materials/Grass_Countryside.mdl|NVIDIA/Materials/Base/Natural/Grass_Countryside.mdl|391bf8ef17fce3f7c18a24871a12f58a776adb12ff43c7f1be64cfb135d083ff"
"Materials/Grass_Countryside/Grass_Countryside_BaseColor.png|NVIDIA/Materials/Base/Natural/Grass_Countryside/Grass_Countryside_BaseColor.png|9a3e1188aba15180d8b46d4bb7673f1ad36704f8ab68739d9774e7cff7da17c0"
"Materials/Grass_Countryside/Grass_Countryside_H.png|NVIDIA/Materials/Base/Natural/Grass_Countryside/Grass_Countryside_H.png|00d800c9203c144e3fc0da0ecc9dd608e526d4525071b5ea435246c1c668f037"
"Materials/Grass_Countryside/Grass_Countryside_Normal.png|NVIDIA/Materials/Base/Natural/Grass_Countryside/Grass_Countryside_Normal.png|26e20bd93b3caa3cb7ca9653137fe3e8166bbd55359584f3aec2b505dcc1bbd3"
"Materials/Grass_Countryside/Grass_Countryside_ORM.png|NVIDIA/Materials/Base/Natural/Grass_Countryside/Grass_Countryside_ORM.png|78a299b2f5806c48d14ca0978fe45e5009e3dbea1507a11f8d463d509e32492f"
)

for entry in "${FILES[@]}"; do
    IFS='|' read -r local remote sha <<< "${entry}"
    target="${DEST}/${local}"
    if [ -f "${target}" ] && echo "${sha}  ${target}" | sha256sum --check --status; then
        echo "ok      ${local}"
        continue
    fi
    mkdir -p "$(dirname "${target}")"
    curl --fail --silent --show-error --output "${target}.part" "${BASE}/${remote}"
    if ! echo "${sha}  ${target}.part" | sha256sum --check --status; then
        echo "error: sha256 mismatch for ${local}" >&2
        rm -f "${target}.part"
        exit 1
    fi
    mv "${target}.part" "${target}"
    echo "fetched ${local}"
done
