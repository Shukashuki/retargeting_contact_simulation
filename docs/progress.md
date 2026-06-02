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
- [x] `ik.py --robot-type wuji_hand --embodiment-type bimanual` → `trajectory_ikrollout.npz` generated, objects move as expected
- [x] `examples/config/override/oakink_wuji.yaml`: safe noise scales for Wuji Hand
  - `first_ctrl_noise_scale: 0.2` (down from 2.0)
  - `njmax_per_env: 512` (up from 350)

### Physics optimization (`run_mjwp.py`)
- [x] `njmax_per_env=512` resolves nefc overflow
- 🔄 NaN rate to be confirmed with `+override=oakink_wuji` (expected <10%)

**Pitfalls encountered:**
- `bimanual.xml` via `<include>` ignores per-file meshdir → moved all STLs to single flat `assets/` dir
- `track_hand_*_palm` site has no matching `ref_hand_*_palm` body in `generate_xml.py` → removed
- Forearm link `mass=0.01` caused rank-deficient Hessian → increased to 0.1, diaginertia to 0.01

---

## Phase 3: HDF5 Contact Force Recording ⬜

- [ ] Test `src/record_contact_hdf5.py` (skeleton exists, untested)
- [ ] Build Wuji Hand + object MuJoCo scene XML
- [ ] Record one HDF5 per path (wuji-retargeting and SPIDER)

---

## Phase 4: Demo Visualization ⬜

- [ ] MuJoCo headless rendering (PYOPENGL_PLATFORM=osmesa for WSL2)
- [ ] `src/viz_heatmap.py` — generate heatmap video from HDF5 sensor data
- [ ] ffmpeg 3-column merge: OakInk RGB | MuJoCo Wuji | heatmap

---

## Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| Confirm run_mjwp.py NaN rate | High | Check log after `+override=oakink_wuji` run |
| Download program.tar (for oakinkv2.py) | Medium | Enables precise grasp-window cropping |
| Test record_contact_hdf5.py | Medium | Requires trajectory_mjwp.npz first |
| Verify left.xml FK orientation | Medium | Left-hand mirrored kinematics not yet visually confirmed |
| Demo video assembly | Low | After Phase 3 complete |
