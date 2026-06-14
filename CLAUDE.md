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

## Pipeline A: OakInk (v1) → wuji-retargeting

```bash
cd /home/roy/retargeting_contact_simulation
/home/roy/miniconda3/envs/retarget_sim/bin/python src/oakink_to_wuji.py \
  --pkl /home/roy/externalTool/OakInk-v2-hub/anno_preview/<seq>.pkl \
  --maniptrans <task_bimanual.pkl> \
  --config configs/wuji_retarget_right.yaml \
  --output outputs/wuji_qpos.npy
```

Output: `(T, 40)` numpy array — right 20 DOF + left 20 DOF, range ≈ [-0.49, 1.63] rad.

## Pipeline D: OakInk-v2 → wuji-retargeting (geometric, no physics)

Geometric retargeting only — MANO FK → wuji-retargeting IK for fingers, wrist from MANO tsl.
No physics simulation. Use `--start-frame` / `--end-frame` (subsampled indices) to match the
same crop window as Pipeline C (find indices with `np.searchsorted` on `mocap_frame_id_list[::4]`).

```bash
cd /home/roy/retargeting_contact_simulation
/home/roy/miniconda3/envs/retarget_sim/bin/python src/oakink_to_wuji.py \
  --pkl /home/roy/externalTool/OakInk-v2-hub/anno_preview/<seq>.pkl \
  --config configs/wuji_retarget_right.yaml \
  --output outputs/<task>_pipeline_d.npy \
  --start-frame <i0> --end-frame <i1> \
  --scene-obj-ids <obj_id_rh> [<obj_id_lh>]
```

Output: `(T, 66)` array — `[wrist6_R | fingers20_R | wrist6_L | fingers20_L | obj7_cap | obj7_bottle]`

Completed: `outputs/unscrew_bottle_pipeline_a.npy` (frames 271:391, 120 frames)

### Comparing Pipeline D vs Pipeline C

```bash
/home/roy/miniconda3/envs/retarget_sim/bin/python src/compare_pipelines_dc.py \
  --task unscrew_bottle
```

## Pipeline B: OakInk (v1) → SPIDER → Wuji Hand

Uses pre-processed maniptrans pkl files. Tasks available in
`example_datasets/raw/oakink/*_bimanual.pkl`.

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

## Pipeline C: OakInk-v2 → SPIDER → Wuji Hand

For sequences from OakInk-v2 Hub. No maniptrans pkl needed — pass `--seq-token` directly.

Find sequences:
```bash
python -c "
import json
data = json.load(open('/home/roy/externalTool/OakInk-v2-hub/program/task_target.json'))
for k, v in data.items():
    print(k, '->', v['task_description'])
" | grep -i <keyword>
```

### Step 0: One-time setup per new object ID

```bash
cd third_party/spider
OBJ=<obj_id>   # e.g. O02@0015@00022  (use @ separators)
OBJ_SAFE=${OBJ//@/_}   # e.g. O02_0015_00022  (used in file paths)

# symlink convex decomposition from oakink assets (same physical object)
ln -sf /home/roy/externalTool/data/oakink/assets/objects/${OBJ}/convex \
    example_datasets/processed/oakinkv2/assets/objects/${OBJ_SAFE}/convex
```

### Step 1: Process raw sequence → trajectory_keypoints.npz

```bash
cd third_party/spider

uv run spider/process_datasets/oakinkv2.py \
  --dataset-dir example_datasets \
  --oakink2-prefix example_datasets/raw/oakinkv2 \
  --seq-token "<seq_token>" \
  --task <task> \
  --mano-assets-root /home/roy/externalTool/data/OakInk2/asset/mano_v1_2 \
  --no-show-viewer
```

Then **manually add `right_object_convex_dir`** to the generated `task_info.json`:
```bash
# task_info.json is at:
# example_datasets/processed/oakinkv2/mano/right/<task>/task_info.json
# Add this key next to right_object_mesh_dir:
#   "right_object_convex_dir": "processed/oakinkv2/assets/objects/<OBJ_SAFE>/convex",
```

### Step 2: Generate scene XML

生成標準 scene.xml（用於一般優化 + knuckle reward）：

```bash
uv run spider/preprocess/generate_xml.py \
  --dataset-name oakinkv2 --task <task> --data-id 0 \
  --embodiment-type right --robot-type wuji_hand
```

若要跑 **contact guidance**（物件有 6 個單軸關節 + 位置作動器），額外生成 scene_act.xml：

```bash
uv run spider/preprocess/generate_xml.py \
  --dataset-name oakinkv2 --task <task> --data-id 0 \
  --embodiment-type right --robot-type wuji_hand \
  --act-scene --free-rot-z   # --free-rot-z: cap 的 rot_z 無作動器（可轉動從動件）
```

兩個指令互不覆蓋：第一個只寫 `scene.xml`，第二個（`--act-scene`）只寫 `scene_act.xml`。

### Step 3: IK

```bash
uv run spider/preprocess/ik.py \
  --dataset-name oakinkv2 --task <task> --data-id 0 \
  --embodiment-type right --robot-type wuji_hand --open-hand --no-show-viewer
```

> **IK 輸出的差異（Wuji Hand 客製化）**
>
> IK 自動追蹤所有 `track_*` site 並將世界座標存入 `contact_pos`。
> 由於我們在 `right.xml` 每個 `fingerX_link3`（DIP 關節體）加了
> `track_hand_right_{finger}_knuckle` site，現在的 `contact_pos` 有 **15 個 mocap entry** 而非原本的 10 個：
>
> | mocap index | body | 說明 |
> |-------------|------|------|
> | 0,3,6,9,12  | `ref_object_right_{finger}_tip` | 物件上的接觸點（contact site） |
> | 1,4,7,10,13 | `ref_hand_right_{finger}_tip`   | 指尖世界座標（MANO fingertip） |
> | 2,5,8,11,14 | `ref_hand_right_{finger}_knuckle` | **新增** DIP 關節世界座標 |
>
> `contact_pos` shape：**(T, 15, 3)**（舊：10, 新：15）
>
> **IK 優化目標不受影響**——knuckle site 只是被動記錄位置，不參與 IK 收斂計算。
> 若用舊的 `contact_pos (T, 10, 3)` 並開啟 `cap_dist_rew_scale > 0`，索引會越界；
> 其他標準 reward 不受影響。

### Step 4: Physics optimization

**Mode A — 標準（qpos tracking）：**
```bash
uv run examples/run_mjwp.py \
  +override=oakinkv2_wuji task=<task> data_id=0 viewer=""
```

**Mode B — Contact Guidance（需先生成 scene_act.xml）：**
```bash
uv run examples/run_mjwp.py \
  +override=oakinkv2_wuji_act task=<task> data_id=0 viewer=""
# override: examples/config/override/oakinkv2_wuji_act.yaml
# 使用 scene_act.xml；cap rot_z 無作動器（可自由旋轉）
# guidance 強度每次迭代衰減 × 0.7，末迭代趨近於 0
```

**Mode C — Fingertip + Knuckle 幾何 reward（需先完成 Step 2+3 產生 contact_pos (T,15,3)）：**
```bash
uv run examples/run_mjwp.py \
  +override=oakinkv2_wuji_cap_contact task=<task> data_id=0 viewer=""
# override: examples/config/override/oakinkv2_wuji_cap_contact.yaml
# contact_guidance: false
# 4 項 reward（各 scale=1.0）：
#   cap_dist_rew   — |dist(fingertip→obj)_sim - dist(fingertip→obj)_ref| per finger
#   cap_dir_rew    — ||unit(fingertip-obj)_sim - unit(fingertip-obj)_ref|| per finger
#   knuckle_dist_rew — 同上但用 DIP 關節位置
#   knuckle_dir_rew  — 同上但用 DIP 關節位置
```

Output: `example_datasets/processed/oakinkv2/wuji_hand/right/<task>/0/trajectory_mjwp.npz`

### Completed tasks

| task | seq_token | obj_id |
|------|-----------|--------|
| unscrew_bottle | `scene_01__A004++seq__670845b55c2609fd17de__2023-04-28-18-37-09` | O02@0015@00022 |

### unscrew_bottle 實驗軌跡（已完成）

存放於 `example_datasets/processed/oakinkv2/wuji_hand/right/unscrew_bottle/0/`：

| 檔案 | Mode | 說明 |
|------|------|------|
| `trajectory_kinematic.npz` | MANO IK ref | qpos (T,33) + contact_pos (T,15,3)；scene.xml 的 IK 解 |
| `trajectory_kinematic_act.npz` | act scene IK ref | qpos (T,32)，contact_pos (T,15,3)；scene_act.xml 的 IK 解，與上面 robot joint angle 可差達 123°，但 fingertip 世界座標差 <2cm（不同 local optimum） |
| `trajectory_mjwp_no_guidance.npz` | Mode A | 標準 qpos tracking，無 contact guidance |
| `trajectory_mjwp_act.npz` | Mode B | contact guidance + free rot_z（已重新生成，2026-06-14） |
| `trajectory_mjwp_cap_contact.npz` | Mode C | fingertip + knuckle 幾何 reward |

> **Act-scene IK 的 local optimum 差異**
> `ik.py --act-scene` 使用 scene_act.xml（object 為 6 個 actuated 單軸 joint）進行優化，
> 因 constraint landscape 不同，optimizer 收斂到與 `scene.xml` IK 完全不同的 joint 解。
> 兩個解均滿足 IK 目標（fingertip 位置接近），但 robot finger joint 構型不同。這是正常行為。

### 比較影片

| 檔案 | 內容 |
|------|------|
| `outputs/modes_comparison.mp4` | **主要比較影片**：3 行 × 2 欄（ref \| sim），每行一個 Mode（A/B/C） |
| `outputs/compare_4cond.mp4` | 4-panel 含指尖軌跡 trail（MANO IK / Mode A / Mode B / Mode C） |
| `outputs/compare_ik.mp4` | 兩欄：standard IK vs act-scene IK，說明兩個 IK 解的差異 |

### 比較影片 render 腳本

| 腳本 | 說明 |
|------|------|
| `src/render_modes.py` | modes_comparison.mp4，Mode A/B/C 各一對 ref\|sim |
| `src/render_4panel.py` | 4-panel 含指尖 trail overlay |
| `src/render_ik_compare.py` | standard IK vs act-scene IK 2-panel 比較 |

## Known Issues

### SPIDER noise scale
`oakink.yaml` sets `first_ctrl_noise_scale=2.0`, which exceeds finger joint limits
(±1.57 rad) and causes ~50% NaN samples. **Always use `+override=oakink_wuji`.**

### chumpy Python 3.12 compatibility
Two patches required in spider venv's chumpy:

1. `inspect.getargspec` removed in 3.12 → patched to `getfullargspec` in `ch.py`
2. `from numpy import bool, int, float, ...` removed in NumPy 2.x → patched to `from numpy import nan, inf` in `__init__.py`

Both already applied to:
- `third_party/spider/.venv/lib/python3.12/site-packages/chumpy/ch.py`
- `third_party/spider/.venv/lib/python3.12/site-packages/chumpy/__init__.py`

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
| SPIDER processed (Wuji, oakink v1) | `third_party/spider/example_datasets/processed/oakink/wuji_hand/` |
| SPIDER processed (Wuji, oakinkv2) | `third_party/spider/example_datasets/processed/oakinkv2/wuji_hand/` |
| OakInk v2 program info | `/home/roy/externalTool/OakInk-v2-hub/program/` |
| Wuji Hand MJCF | `third_party/wuji-retargeting/.../wuji_hand_description/mjcf/` |
