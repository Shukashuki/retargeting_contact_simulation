# retargeting_contact_simulation

OakInk v2 human hand demonstrations → dual-path retargeting → Wuji Hand simulation and contact force recording.

## Repository Structure

```
retargeting_contact_simulation/
├── third_party/
│   ├── spider/              # Shukashuki/spider fork (facebookresearch/spider + Wuji Hand)
│   └── wuji-retargeting/    # wuji-technology/wuji-retargeting
├── src/
│   ├── oakink_to_wuji.py       # OakInk MANO FK → wuji-retargeting → Wuji Hand qpos
│   ├── record_contact_hdf5.py  # MuJoCo replay + HDF5 contact force recording
│   └── viz_heatmap.py          # Fingertip touch sensor heatmap → mp4
├── configs/
│   ├── wuji_retarget_right.yaml
│   └── wuji_retarget_left.yaml
└── data/                    # Symlinks (.gitignore), not committed
    ├── oakink2 → /home/roy/externalTool/data/OakInk2
    └── oakink  → /home/roy/externalTool/data/oakink
```

## Environments

| Purpose | Python env | Command |
|---------|-----------|---------|
| src/ scripts | retarget_sim conda | `/home/roy/miniconda3/envs/retarget_sim/bin/python` |
| SPIDER pipeline | spider conda (uv) | `cd third_party/spider && uv run ...` |
| MANO FK only | OakInk2 conda | `/home/roy/externalTool/data/OakInk2/.conda/bin/python` |

## Pipeline A: OakInk → wuji-retargeting

```bash
cd /home/roy/retargeting_contact_simulation
/home/roy/miniconda3/envs/retarget_sim/bin/python src/oakink_to_wuji.py \
  --pkl /home/roy/externalTool/OakInk-v2-hub/anno_preview/<seq>.pkl \
  --config configs/wuji_retarget_right.yaml \
  --output outputs/wuji_qpos.npy
```

Output: `(T, 40)` numpy array — right 20 DOF + left 20 DOF, range ≈ [-0.49, 1.63] rad.

## Pipeline B: OakInk → SPIDER → Wuji Hand

```bash
cd third_party/spider

# 1. Generate scene XML
uv run spider/preprocess/generate_xml.py \
  --dataset-name oakink --task <task> --data-id <id> \
  --embodiment-type bimanual --robot-type wuji_hand

# 2. IK
uv run spider/preprocess/ik.py \
  --dataset-name oakink --task <task> --data-id <id> \
  --embodiment-type bimanual --robot-type wuji_hand --open-hand

# 3. Physics optimization — must use oakink_wuji, NOT oakink
uv run examples/run_mjwp.py \
  +override=oakink_wuji task=<task> data_id=<id> viewer=""
```

Output: `example_datasets/processed/oakink/wuji_hand/bimanual/<task>/<id>/trajectory_mjwp.npz`

## Known Issues

### SPIDER noise scale
`oakink.yaml` sets `first_ctrl_noise_scale=2.0`, which exceeds finger joint limits
(±1.57 rad) and causes ~50% NaN samples. **Always use `+override=oakink_wuji`.**

### chumpy Python 3.11 compatibility
`inspect.getargspec` removed in Python 3.11. Already patched in:
`/home/roy/externalTool/data/OakInk2/thirdparty/chumpy/chumpy/ch.py`
(changed to `inspect.getfullargspec`).

### Wuji Hand bimanual XML
- `right.xml`: no prefix on body names
- `left.xml`: all bodies/joints prefixed with `l_` to avoid name conflicts
- Left-hand STL files prefixed with `l_` in the shared `assets/` directory
- Each palm has a 6DOF forearm chain (3 slides + 3 revolutes, kp=1000/200)

### SPIDER njmax
52-DOF bimanual exceeds default njmax=350. Use `njmax_per_env=512` (already in `oakink_wuji.yaml`).

## Data Locations

| Data | Path |
|------|------|
| OakInk v2 anno_preview pkl | `/home/roy/externalTool/OakInk-v2-hub/anno_preview/` |
| OakInk v2 raw images | `/home/roy/externalTool/OakInk-v2-hub/data/` |
| OakInk MANO processed | `/home/roy/externalTool/data/oakink/mano/bimanual/` |
| SPIDER processed (Allegro) | `/home/roy/externalTool/data/oakink/allegro/bimanual/` |
| SPIDER processed (Wuji) | `third_party/spider/example_datasets/processed/oakink/wuji_hand/` |
| Wuji Hand MJCF | `third_party/wuji-retargeting/.../wuji_hand_description/mjcf/` |
