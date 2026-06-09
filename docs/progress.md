# Project Progress

## Goal
Retarget OakInk v2 human hand demonstrations to Wuji Hand via two paths (SPIDER and wuji-retargeting), replay in MuJoCo, record contact forces, and produce a comparison demo video.

---

## Phase 0: Monorepo Setup ✅

- [x] Create `retargeting_contact_simulation` GitHub repo
- [x] Fork `facebookresearch/spider` → `Shukashuki/spider`
- [x] Add git submodules: spider + wuji-retargeting
- [x] Create `src/` skeleton (oakink_to_wuji.py, record_contact_hdf5.py, viz_heatmap.py)
- [x] Create `configs/` for wuji-retargeting right/left hand
- [x] Update SPIDER to latest upstream (includes oakinkv2 pipeline commit `b2e60f0`)

**Key decision:** SPIDER's `oakinkv2.py` requires `program.tar` (program_info JSON) for grasp primitive timing, which is not downloaded locally. Using the older `oakink.py` pipeline (maniptrans pkl) instead — processed data already exists for 5 tasks.

---

## Phase 1: OakInk → wuji-retargeting ✅

- [x] `src/oakink_to_wuji.py`: anno_preview pkl → MANO FK → wuji-retargeting → qpos
- [x] Fix chumpy Python 3.11 compatibility (`getargspec` → `getfullargspec`)
- [x] Remove `os.chdir()` side effect, use absolute paths throughout
- [x] Build `retarget_sim` conda env (Python 3.11, pinocchio 3.8.0, mujoco 3.9.0, torch 2.5.1+cu121)

**Verified:** Output shape `(2613, 40)`, range `[-0.49, 1.63]` rad — within Wuji Hand joint limits.

**Coordinate system note:** wuji-retargeting normalizes to wrist-relative internally, so MANO world-space joints can be fed directly. Adjust `mediapipe_rotation` in config if axis orientation is off.

---

## Phase 2: SPIDER + Wuji Hand ✅ (IK) / 🔄 (physics tuning)

### Adding Wuji Hand to SPIDER
- [x] `spider/assets/robots/wuji_hand/right.xml` with:
  - 6DOF forearm chain (3 slides + 3 revolutes) for world-space wrist positioning
  - SPIDER-required site names (`right_palm`, `right_{thumb,index,middle,ring,pinky}_tip`)
  - Only fingertip track/trace sites (no palm track/trace — no matching `ref_hand` body exists)
  - `right_groundplane` material (required by `generate_xml.py`)
- [x] `spider/assets/robots/wuji_hand/left.xml`: same with `l_` prefix on all body/joint names
- [x] `spider/assets/robots/wuji_hand/bimanual.xml`: include right + left, nq=52 nu=52
- [x] Flat `assets/` directory: right-hand STLs as-is, left-hand STLs with `l_` prefix
- [x] Sync to `example_datasets/processed/oakink/assets/robots/wuji_hand/`

### SPIDER Pipeline
- [x] `generate_xml.py --robot-type wuji_hand --embodiment-type bimanual` → scene XML OK
- [x] `ik.py --robot-type wuji_hand --embodiment-type bimanual` → `trajectory_ikrollout.npz` generated
- [x] `examples/config/override/oakink_wuji.yaml`: safe noise scales for Wuji Hand
  - `first_ctrl_noise_scale: 0.2` (down from 2.0)
  - `njmax_per_env: 512` (up from 350)

### Contact pairs
- [x] Added 5 `collision_hand_right_*` sphere geoms (size=0.007m) at right fingertips in `right.xml`
- [x] Added 5 `collision_hand_left_*` sphere geoms at left fingertips in `left.xml`
- [x] `generate_xml.py` re-run → scene.xml with **199 contact pairs** (floor-object + hand-object)
- [x] 10 fingertip `<touch>` sensors re-added to scene.xml after regeneration

### IK fixes (`ik.py`)
- [x] Zero object free-joint velocities in `qvel_list` (IK finite-diff produces ~3 rad/s artifact)
- [x] SPIDER-style warm-up reset: `qpos/qvel ← qpos_list[0]` + `mj_forward` after warm-up `mj_step`
- [x] Rollout contact softening: `impratio=1.0`, `integrator=Euler` (scene default is `impratio=10, RK4`)

### Physics optimization (`run_mjwp.py`)
- [x] `njmax_per_env=512` resolves nefc overflow
- [x] `trajectory_mjwp.npz` generated (5 ticks × 40 steps = 200 frames)

**Pitfalls encountered:**
- `bimanual.xml` via `<include>` ignores per-file meshdir → moved all STLs to single flat `assets/` dir
- `track_hand_*_palm` site has no matching `ref_hand_*_palm` body in `generate_xml.py` → removed
- Forearm link `mass=0.01` caused rank-deficient Hessian → increased to 0.1, diaginertia to 0.01
- `ikrollout` explosion: warm-up `mj_step` impulse + `impratio=10` → fix: reset after warm-up + soften contacts
- Object geoms have `contype=0/conaffinity=0` → only collide via explicit contact pairs (floor-object pairs required)

---

## Phase 3: HDF5 Contact Force Recording ✅

- [x] `src/record_contact_hdf5.py`: kinematic mode (`mj_forward`) + physics mode (`--physics`, `mj_step`)
  - Handles 2D `(T, nq)` npz format (kinematic/ikrollout) and 3D mjwp format
- [x] SPIDER mjwp: `outputs/recordings_spider/uncap_mjwp.h5` — 189/200 frames with sensor > 0
- [x] Wuji physics: `outputs/recordings_wuji/uncap_physics.h5` — 2/62 frames (hand never reaches objects)

---

## Phase 4: Demo Visualization ✅

- [x] `src/render_scene.py`: kinematic mode + physics mode (`--physics`, `mj_step` with PD control)
- [x] Physics-correct renders:
  - `outputs/spider/wuji_hand_uncap_mjwp.mp4` — SPIDER mjwp (200 frames, physics-optimized)
  - `outputs/wuji_retargeting/wuji_hand_uncap_physics.mp4` — Wuji physics (62 frames, objects free)
- [x] Heatmap videos:
  - `outputs/spider/heatmap_mjwp.mp4`
  - `outputs/wuji_retargeting/heatmap_physics.mp4`
- [x] `outputs/demo_comparison.mp4` — 2×2 grid (1280×960, 200 frames, 30fps)

**Physics analysis findings:**
- `trajectory_kinematic.npz`: pure IK, object positions scripted from OakInk reference (not physics)
- `trajectory_mjwp.npz`: SPIDER physics-optimized — objects stationary (cap Δz=3mm), hand-object dist 15–20cm → optimization has not fully converged for Wuji Hand
- `wuji_qpos_uncap.npy`: wuji-retarget IK only, hand 16–20cm from objects throughout, objects scripted
- In physics mode (`mj_step`), wuji objects are stationary — confirms wuji hand never contacts objects

**`--physics` mode (both scripts):**
- Init: `qpos[:nq] = traj[0]`, `qvel = 0`
- Per frame: `ctrl[:nu] = traj[i, :nu]` → `mj_step × (dt/sim_dt)` — objects evolve freely

---

## Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| Improve run_mjwp.py convergence for Wuji Hand | High | Hand not reaching objects; tune reward weights or noise scale |
| Verify left.xml FK orientation | Medium | Left-hand mirrored kinematics not yet visually confirmed |
| Download program.tar (for oakinkv2.py) | Low | Enables precise grasp-window cropping |
