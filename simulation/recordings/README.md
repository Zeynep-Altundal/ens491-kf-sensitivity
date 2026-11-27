# recordings

Flights written by `../rviz_drone/rviz_drone/recorder.py` in the common recording format v1
(`<trajectory>_seed<seed>.npz`; format, rates, frames and metadata in `../rviz_drone/README.md`,
section "Flight recorder"). The `.npz` files are generated and not version-controlled; recreate
them with

```bash
cd simulation/rviz_drone && ../../.venv/bin/python -m rviz_drone.recorder --trajectory mission --seed 0
```

Corruption scenarios of a recording (`<recording>__<scenario>.npz`, also not version-controlled)
are written by `../rviz_drone/rviz_drone/corruption.py` (scenario table in `../rviz_drone/README.md`,
section "Corruption scenarios"):

```bash
cd simulation/rviz_drone && ../../.venv/bin/python -m rviz_drone.corruption
```
