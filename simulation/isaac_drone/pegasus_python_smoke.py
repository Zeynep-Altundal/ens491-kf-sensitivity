"""Headless flight check of Pegasus Simulator with its Python control backend (no PX4, no ROS).

Based on examples/4_python_single_vehicle.py of Pegasus Simulator v5.1.0, with three changes:
headless rendering, a fixed duration instead of an endless loop, and a second backend that records
the ground-truth state and every sensor sample. The example controller (NonlinearController from
examples/utils) is the first backend and therefore the one that drives the rotors.

Run with the Python of Isaac Sim 5.1.0:
    source simulation/isaac_drone/env.sh
    "$ISAACSIM_PYTHON" simulation/isaac_drone/pegasus_python_smoke.py --out /path/to/flight.npz

Output: one .npz with the state per physics step and the samples of each sensor, plus SMOKE lines.
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="output .npz file")
parser.add_argument("--duration", type=float, default=20.0, help="simulated time [s]")
parser.add_argument("--env", default="Curved Gridroom", help="Pegasus environment name")
parser.add_argument("--pegasus", default=os.environ.get("PEGASUS_PATH", os.path.expanduser("~/PegasusSimulator")))
args = parser.parse_args()

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})
print(f"SMOKE app_started seconds={time.time() - t0:.1f}", flush=True)

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from omni.isaac.core.world import World  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS  # noqa: E402

examples = os.path.join(args.pegasus, "examples")
sys.path.insert(0, os.path.join(examples, "utils"))
from nonlinear_controller import NonlinearController  # noqa: E402


class Recorder(Backend):
    """Backend that only records; it is never the first backend, so it never commands the rotors."""

    def __init__(self):
        super().__init__(BackendConfig())
        self.t = 0.0
        self.state_rows = []
        self.sensors = {}

    def update_state(self, state):
        self.state_rows.append(np.concatenate([[self.t], state.position, state.attitude,
                                               state.linear_velocity, state.linear_body_velocity,
                                               state.angular_velocity, state.linear_acceleration]))

    def update_sensor(self, sensor_type, data):
        flat = {}
        for k, v in data.items():
            a = np.atleast_1d(np.asarray(v, dtype=float)) if not isinstance(v, str) else None
            if a is not None:
                flat[k] = a
        self.sensors.setdefault(sensor_type, []).append((self.t, flat))

    def input_reference(self):
        return []

    def update(self, dt):
        self.t += dt

    def start(self):
        self.t = 0.0

    def stop(self):
        pass

    def reset(self):
        pass


timeline = omni.timeline.get_timeline_interface()
pg = PegasusInterface()
pg._world = World(**pg._world_settings)
world = pg.world
print(f"SMOKE world_settings={pg._world_settings}", flush=True)
pg.load_environment(SIMULATION_ENVIRONMENTS[args.env])
print(f"SMOKE environment_loaded name={args.env!r} seconds={time.time() - t0:.1f}", flush=True)

recorder = Recorder()
config = MultirotorConfig()
config.backends = [
    NonlinearController(trajectory_file=os.path.join(examples, "trajectories", "pitch_relay_90_deg_2.csv"),
                        results_file=None, Ki=[0.5, 0.5, 0.5], Kr=[2.0, 2.0, 2.0]),
    recorder,
]
init_pos = [2.3, -1.5, 0.07]
Multirotor("/World/quadrotor1", ROBOTS["Iris"], 0, init_pos,
           Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(), config=config)
print(f"SMOKE sensors={[type(s).__name__ for s in config.sensors]}", flush=True)
world.reset()

timeline.play()
t1 = time.time()
steps = 0
while simulation_app.is_running() and recorder.t < args.duration:
    world.step(render=True)
    steps += 1
wall = time.time() - t1
timeline.stop()

state = np.array(recorder.state_rows)
out = {"state": state,
       "state_columns": np.array(["t", "px", "py", "pz", "qx", "qy", "qz", "qw", "vx", "vy", "vz",
                                  "vbx", "vby", "vbz", "wx", "wy", "wz", "ax", "ay", "az"])}
for name, samples in recorder.sensors.items():
    out[f"{name}_t"] = np.array([s[0] for s in samples])
    for key in samples[0][1]:
        out[f"{name}_{key}"] = np.array([s[1][key] for s in samples])
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, **out)

print(f"SMOKE steps={steps} sim_time={recorder.t:.3f} wall={wall:.1f} state_rows={len(state)}", flush=True)
for name, samples in recorder.sensors.items():
    print(f"SMOKE sensor {name}: samples={len(samples)} keys={sorted(samples[0][1])}", flush=True)
if len(state):
    for target in (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, args.duration):
        i = min(int(np.searchsorted(state[:, 0], target)), len(state) - 1)
        print(f"SMOKE position t={state[i, 0]:6.2f} p=[{state[i, 1]:+.3f}, {state[i, 2]:+.3f}, {state[i, 3]:+.3f}]",
              flush=True)
    d = np.linalg.norm(state[:, 1:4] - state[0, 1:4], axis=1)
    print(f"SMOKE max_displacement_m={d.max():.3f} final_displacement_m={d[-1]:.3f} "
          f"z_range=[{state[:, 3].min():.3f}, {state[:, 3].max():.3f}]", flush=True)
print(f"SMOKE wrote {args.out}", flush=True)

simulation_app.close()
