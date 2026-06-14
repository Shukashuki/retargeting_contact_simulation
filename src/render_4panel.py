"""Render 4-panel comparison video with fingertip IK trajectory trails.

Panels: MANO IK (ref) | No Contact Guidance | Contact Guidance | Geometric Reward

Each panel shows:
  - Colored trail: thumb (red) / index (green) / middle (blue) fingertip paths
  - Gray dots on panels 1-3: MANO IK reference path for comparison
  - Bright large dot at current frame position

Usage:
  python src/render_4panel.py --out outputs/compare_4cond.mp4
"""

from __future__ import annotations

import argparse
import os

import cv2
import imageio
import mujoco
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────────────
SPIDER   = os.path.join(os.path.dirname(__file__), "../third_party/spider")
RUN_DIR  = os.path.join(SPIDER, "example_datasets/processed/oakinkv2/wuji_hand/right/unscrew_bottle/0")
SCENE     = os.path.join(RUN_DIR, "../scene.xml")
SCENE_ACT = os.path.join(RUN_DIR, "../scene_act.xml")

PANELS = [
    {
        "label": "MANO IK (reference)",
        "sublabel": "IK fingertip path",
        "scene": SCENE,
        "npz":   os.path.join(RUN_DIR, "trajectory_kinematic.npz"),
        "flat":  True,   # (T, nq) — not ticks × ctrl_steps
    },
    {
        "label": "No Contact Guidance",
        "sublabel": "qpos tracking only",
        "scene": SCENE,
        "npz":   os.path.join(RUN_DIR, "trajectory_mjwp_no_guidance.npz"),
        "flat":  False,
    },
    {
        "label": "Contact Guidance",
        "sublabel": "act scene  free rot-z",
        "scene": SCENE_ACT,
        "npz":   os.path.join(RUN_DIR, "trajectory_mjwp_act.npz"),
        "flat":  False,
    },
    {
        "label": "Geometric Reward",
        "sublabel": "fingertip+knuckle dist+dir",
        "scene": SCENE,
        "npz":   os.path.join(RUN_DIR, "trajectory_mjwp_cap_contact.npz"),
        "flat":  False,
    },
]

# ─── render settings ──────────────────────────────────────────────────────────
W, H          = 480, 480
FPS           = 25              # frames per second of output video
RENDER_EVERY  = 2               # sim steps per rendered frame
N_FRAMES      = 200

# ─── trail settings ───────────────────────────────────────────────────────────
TRAIL_LEN     = 60              # show last N frames as trail
TRAIL_STRIDE  = 3               # sample every Nth frame in trail (reduces geom count)
REF_STRIDE    = 6               # subsample reference overlay
REF_ALPHA_MAX = 0.45            # reference overlay max alpha

# ─── finger definitions (name, site, BGR for OpenCV, RGB float for MuJoCo) ───
FINGERS = [
    {"name": "thumb",  "site": "track_hand_right_thumb_tip",  "rgb": (1.0, 0.25, 0.2),  "bgr": (50,  64, 255)},
    {"name": "index",  "site": "track_hand_right_index_tip",  "rgb": (0.2, 0.9,  0.2),  "bgr": (50, 230, 50)},
    {"name": "middle", "site": "track_hand_right_middle_tip", "rgb": (0.3, 0.55, 1.0),  "bgr": (255, 140, 75)},
]
REF_RGB = (0.9, 0.9, 0.9)

CAP_LOOKAT = np.array([0.036, -0.173, -0.041], dtype=np.float64)


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_model_data(scene_path: str):
    m = mujoco.MjModel.from_xml_path(scene_path)
    m.vis.global_.offwidth  = W
    m.vis.global_.offheight = H
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=H, width=W)
    return m, d, r


def get_site_ids(model) -> list[int]:
    ids = []
    for f in FINGERS:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f["site"])
        if sid < 0:
            raise RuntimeError(f"Site not found: {f['site']}")
        ids.append(sid)
    return ids


def precompute_traj(frames: list[np.ndarray], model, data, site_ids: list[int]) -> np.ndarray:
    """Replay frames and record fingertip world positions. Returns (N_FRAMES, n_fingers, 3)."""
    positions = []
    for qpos in frames:
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        pos = np.stack([data.site_xpos[sid].copy() for sid in site_ids])
        positions.append(pos)
    mujoco.mj_resetData(model, data)
    return np.array(positions)  # (N, n_fingers, 3)


def _add_sphere(scene, pos, rgb, alpha: float, radius: float):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    r, gv, b = rgb
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [radius, radius, radius],
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        [float(r), float(gv), float(b), float(alpha)],
    )
    scene.ngeom += 1


def draw_trails(scene, panel_traj: np.ndarray, ref_traj: np.ndarray | None, fi: int, is_ref: bool):
    """Add trail spheres to the MuJoCo scene (call after update_scene)."""
    n_fingers = len(FINGERS)

    # ── reference MANO IK overlay (gray, fixed for full video) ──
    if not is_ref and ref_traj is not None:
        for rf in range(0, N_FRAMES, REF_STRIDE):
            is_curr_ref = abs(rf - fi) < REF_STRIDE
            alpha = REF_ALPHA_MAX * (1.2 if is_curr_ref else 0.7)
            radius = 0.004 if is_curr_ref else 0.003
            for j in range(n_fingers):
                _add_sphere(scene, ref_traj[rf, j], REF_RGB, alpha, radius)

    # ── current panel trail (colored, last TRAIL_LEN frames) ──
    trail_start = max(0, fi - TRAIL_LEN)
    for tf in range(trail_start, fi + 1, TRAIL_STRIDE):
        age = fi - tf
        age_frac = age / max(TRAIL_LEN, 1)   # 0=current, 1=oldest visible

        for j, fdef in enumerate(FINGERS):
            pos = panel_traj[tf, j]
            if tf == fi:
                alpha  = 1.0
                radius = 0.009
            else:
                alpha  = 0.15 + 0.75 * (1.0 - age_frac)
                radius = 0.003 + 0.003 * (1.0 - age_frac)
            _add_sphere(scene, pos, fdef["rgb"], alpha, radius)


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 0.35
    cam.azimuth   = -45
    cam.elevation = -20
    cam.lookat[:] = CAP_LOOKAT
    return cam


def render_frame(renderer, model, data, panel_traj, ref_traj, fi: int, site_ids, is_ref: bool) -> np.ndarray:
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    renderer.update_scene(data, camera=make_camera(), scene_option=options)
    draw_trails(renderer.scene, panel_traj, ref_traj, fi, is_ref)
    return renderer.render().copy()


def add_panel_label(img: np.ndarray, label: str, sublabel: str) -> np.ndarray:
    img = img.copy()
    box_h = 52
    cv2.rectangle(img, (0, 0), (W, box_h), (18, 18, 18), -1)
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, sublabel, (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def add_legend_and_frame(img: np.ndarray, fi: int, is_ref: bool) -> np.ndarray:
    img = img.copy()
    lh = 20
    n = len(FINGERS) + (0 if is_ref else 1)   # +1 for ref gray entry
    box_y = H - 8 - n * lh - 4
    cv2.rectangle(img, (0, box_y - 4), (145, H - 4), (18, 18, 18), -1)

    for i, fdef in enumerate(FINGERS):
        y = box_y + 14 + i * lh
        b, gv, r = fdef["bgr"]   # already in BGR
        cv2.circle(img, (14, y - 4), 6, (b, gv, r), -1)
        cv2.putText(img, fdef["name"], (26, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (210, 210, 210), 1, cv2.LINE_AA)

    if not is_ref:
        y = box_y + 14 + len(FINGERS) * lh
        cv2.circle(img, (14, y - 4), 5, (220, 220, 220), -1)
        cv2.putText(img, "MANO ref", (26, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (170, 170, 170), 1, cv2.LINE_AA)

    # frame counter top-right
    txt = f"{fi+1}/{N_FRAMES}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (W - tw - 12, 0), (W, 22), (18, 18, 18), -1)
    cv2.putText(img, txt, (W - tw - 6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return img


# ─── trajectory extraction ────────────────────────────────────────────────────

def extract_flat_frames(qpos: np.ndarray, n: int) -> list[np.ndarray]:
    T = len(qpos)
    return [qpos[i] for i in np.linspace(0, T - 1, n).astype(int)]


def extract_mjwp_frames(qpos: np.ndarray, n: int) -> list[np.ndarray]:
    n_ticks, ctrl_steps, _ = qpos.shape
    frames = []
    for tick in range(n_ticks):
        for step in range(ctrl_steps):
            if step % RENDER_EVERY == 0:
                frames.append(qpos[tick, step])
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


# ─── main ─────────────────────────────────────────────────────────────────────

def main(out: str):
    # ── load panel data ──
    panel_data = []
    for p in PANELS:
        raw = np.load(p["npz"])["qpos"]
        frames = extract_flat_frames(raw, N_FRAMES) if p["flat"] else extract_mjwp_frames(raw, N_FRAMES)
        model, data, renderer = make_model_data(p["scene"])
        site_ids = get_site_ids(model)
        panel_data.append({
            "frames": frames, "model": model, "data": data,
            "renderer": renderer, "label": p["label"], "sublabel": p["sublabel"],
            "site_ids": site_ids,
        })

    # ── precompute fingertip world trajectories ──
    print("Precomputing fingertip trajectories...")
    for i, pd in enumerate(panel_data):
        print(f"  [{i+1}/{len(PANELS)}] {PANELS[i]['label']}")
        pd["traj"] = precompute_traj(pd["frames"], pd["model"], pd["data"], pd["site_ids"])

    ref_traj = panel_data[0]["traj"]   # MANO IK = ground truth overlay

    # ── render ──
    print(f"Rendering {N_FRAMES} frames × {len(PANELS)} panels at {W}x{H}...")
    images = []
    for fi in range(N_FRAMES):
        if fi % 25 == 0:
            print(f"  frame {fi}/{N_FRAMES}")
        row = []
        for i, pd in enumerate(panel_data):
            qpos = pd["frames"][fi]
            pd["data"].qpos[:len(qpos)] = qpos
            mujoco.mj_forward(pd["model"], pd["data"])

            is_ref = (i == 0)
            img = render_frame(
                pd["renderer"], pd["model"], pd["data"],
                pd["traj"], ref_traj, fi, pd["site_ids"], is_ref,
            )
            img = add_panel_label(img, pd["label"], pd["sublabel"])
            img = add_legend_and_frame(img, fi, is_ref)
            row.append(img)
        images.append(np.concatenate(row, axis=1))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, images, fps=FPS)
    print(f"\nSaved → {out}  ({N_FRAMES} frames, {FPS} fps, {len(PANELS)} panels, {W*len(PANELS)}x{H})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/compare_4cond.mp4")
    args = parser.parse_args()
    main(args.out)
