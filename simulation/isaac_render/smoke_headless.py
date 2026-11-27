"""Headless start-up check of Isaac Sim 6.0.1: start SimulationApp, render a small stage, shut down.

Only ``omni.usd`` and ``pxr`` are used, because ``isaacsim.core.api`` is in ``extsDeprecated`` in 6.0.1.

Run with the Python of Isaac Sim (never with the system Python):
    source simulation/isaac_render/env.sh
    "$ISAACSIM_PYTHON" simulation/isaac_render/smoke_headless.py

Prints one line per check, prefixed with SMOKE, so that the result can be found in the long Kit log.
"""

import sys
import time

t0 = time.time()

from isaacsim import SimulationApp  # noqa: E402  (must be imported before any omni module)

simulation_app = SimulationApp({"headless": True})
print(f"SMOKE app_started seconds={time.time() - t0:.1f}", flush=True)

import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

app = omni.kit.app.get_app()
print(f"SMOKE kit_version={app.get_kit_version()}", flush=True)
print(f"SMOKE app_version={app.get_app_version()}", flush=True)

tokens = carb.tokens.get_tokens_interface()
for name in ("home", "data", "cache", "logs", "documents", "temp", "omni_config", "omni_data", "omni_logs",
             "omni_cache", "omni_global_config", "omni_global_data", "omni_global_logs", "omni_global_cache",
             "omni_documents", "shared_documents", "app_documents"):
    print(f"SMOKE token {name}={tokens.resolve('${' + name + '}')}", flush=True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Cube.Define(stage, "/World/Ground").AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.01))
UsdGeom.Cube.Define(stage, "/World/Box").AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(2500.0)
n_updates = 120
t1 = time.time()
for _ in range(n_updates):
    simulation_app.update()
print(f"SMOKE updated updates={n_updates} wall={time.time() - t1:.2f}", flush=True)

simulation_app.close()
print(f"SMOKE closed total_seconds={time.time() - t0:.1f}", flush=True)
sys.exit(0)
