# Tests of the Isaac Sim renderer

`test_isaac_render_pose.py` checks the conversion of the simulator pose (ENU world, FLU body,
Hamilton quaternion) to the USD translate and orient values in `../usd_pose.py`. The tests use
numpy only and run with the repository test suite (`pytest`); Isaac Sim is not needed.
