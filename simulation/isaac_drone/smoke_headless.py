"""Headless start-up check of Isaac Sim: start SimulationApp, step an empty world, shut down.

Run with the Python of Isaac Sim (never with the system Python):
    source simulation/isaac_drone/env.sh
    "$ISAACSIM_PYTHON" simulation/isaac_drone/smoke_headless.py

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
from isaacsim.core.api import World  # noqa: E402

app = omni.kit.app.get_app()
print(f"SMOKE kit_version={app.get_kit_version()}", flush=True)
print(f"SMOKE app_version={app.get_app_version()}", flush=True)

tokens = carb.tokens.get_tokens_interface()
for name in ("home", "data", "cache", "logs", "documents", "shared_data", "app_data", "temp"):
    print(f"SMOKE token {name}={tokens.resolve('${' + name + '}')}", flush=True)

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 250.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()
world.reset()
n_steps = 250
t1 = time.time()
for _ in range(n_steps):
    world.step(render=False)
print(f"SMOKE stepped steps={n_steps} sim_time={world.current_time:.3f} wall={time.time() - t1:.2f}",
      flush=True)

simulation_app.close()
print(f"SMOKE closed total_seconds={time.time() - t0:.1f}", flush=True)
sys.exit(0)
