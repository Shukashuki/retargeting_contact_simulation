"""Compare standard IK vs act-scene IK side by side.

Left:  trajectory_kinematic.npz  replayed in scene.xml     (free-joint object, nq=33)
Right: trajectory_kinematic_act.npz replayed in scene_act.xml (6-DOF joints, nq=32)

Key finding: finger JOINT angles differ by up to 123° even though fingertip world
positions are almost identical (<2 cm).  This explains why the two physics
optimization runs produce different results: they start from fundamentally different
IK reference trajectories.

Usage:
  python src/render_ik_compare.py --out outputs/compare_ik.mp4
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
        "id": "standard",
        "title":    "Standard IK  (scene.xml)",
        "subtitle": "Object: free joint  |  nq=33",
        "note":     "Physics opt: No guidance / Geometric reward",
        "scene":    SCENE,
        "npz":      os.path.join(RUN_DIR, "trajectory_kinematic.npz"),
    },
    {
        "id": "act",
        "title":    "Act-Scene IK  (scene_act.xml)",
        "subtitle": "Object: 6 single-DOF joints  |  nq=32",
        "note":     "Physics opt: Contact guidance (free rot-z)",
        "scene":    SCENE_ACT,
        "npz":      os.path.join(RUN_DIR, "trajectory_kinematic_act.npz"),
    },
]

# ─── render settings ──────────────────────────────────────────────────────────
W, H         = 640, 640
FPS          = 20
N_FRAMES     = 196

# ─── trail ────────────────────────────────────────────────────────────────────
TRAIL_LEN    = 60
TRAIL_STRIDE = 2
CROSS_STRIDE = 4     # stride for drawing the OTHER panel's fingertip path

FINGERS = [
    {"name": "thumb",  "site": "track_hand_right_thumb_tip",  "rgb": (1.0, 0.25, 0.2),  "bgr": (50,  64, 255)},
    {"name": "index",  "site": "track_hand_right_index_tip",  "rgb": (0.2, 0.9,  0.2),  "bgr": (50, 230,  50)},
    {"name": "middle", "site": "track_hand_right_middle_tip", "rgb": (0.3, 0.55, 1.0),  "bgr": (255,140,  75)},
]
CROSS_RGB  = (0.85, 0.85, 0.85)   # gray: fingertip path of the OTHER ik in this panel
CROSS_ALPHA_MAX = 0.5

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
            raise RuntimeError(f"Site not found in model: {f['site']}")
        ids.append(sid)
    return ids


def precompute_traj(frames: list[np.ndarray], model, data, site_ids: list[int]) -> np.ndarray:
    """Replay frames, record fingertip world positions → (N, n_fingers, 3)."""
    out = []
    for qpos in frames:
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        out.append(np.stack([data.site_xpos[sid].copy() for sid in site_ids]))
    mujoco.mj_resetData(model, data)
    return np.array(out)


def _add_sphere(scene, pos, rgb, alpha: float, radius: float):
    if scene.ngeom >= scene.maxgeom:
        return
    r, gv, b = rgb
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [radius, radius, radius],
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        [float(r), float(gv), float(b), float(alpha)],
    )
    scene.ngeom += 1


def draw_trails(scene, my_traj: np.ndarray, other_traj: np.ndarray, fi: int):
    """Add sphere geoms to the scene for both my trail and the cross-overlay."""
    # ── other IK fingertip path (full trajectory, gray) ──
    for rf in range(0, N_FRAMES, CROSS_STRIDE):
        is_near_curr = abs(rf - fi) <= CROSS_STRIDE
        alpha = CROSS_ALPHA_MAX * (1.5 if is_near_curr else 0.6)
        radius = 0.005 if is_near_curr else 0.003
        for j in range(len(FINGERS)):
            _add_sphere(scene, other_traj[rf, j], CROSS_RGB, alpha, radius)

    # ── this panel's own trail (colored, recent N frames) ──
    trail_start = max(0, fi - TRAIL_LEN)
    for tf in range(trail_start, fi + 1, TRAIL_STRIDE):
        age_frac = (fi - tf) / max(TRAIL_LEN, 1)
        for j, fdef in enumerate(FINGERS):
            if tf == fi:
                _add_sphere(scene, my_traj[tf, j], fdef["rgb"], 1.0, 0.010)
            else:
                alpha  = 0.15 + 0.75 * (1.0 - age_frac)
                radius = 0.003 + 0.004 * (1.0 - age_frac)
                _add_sphere(scene, my_traj[tf, j], fdef["rgb"], alpha, radius)


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 0.30      # closer than the 4-panel video
    cam.azimuth   = -45
    cam.elevation = -25
    cam.lookat[:] = CAP_LOOKAT
    return cam


def render_frame(renderer, model, data, my_traj, other_traj, fi: int) -> np.ndarray:
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    renderer.update_scene(data, camera=make_camera(), scene_option=options)
    draw_trails(renderer.scene, my_traj, other_traj, fi)
    return renderer.render().copy()


def add_header(img: np.ndarray, title: str, subtitle: str, note: str) -> np.ndarray:
    img = img.copy()
    box_h = 72
    cv2.rectangle(img, (0, 0), (W, box_h), (18, 18, 18), -1)
    cv2.putText(img, title,    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(img, note,     (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 200, 140), 1, cv2.LINE_AA)
    return img


def add_footer(img: np.ndarray, fi: int, joint_diff: np.ndarray) -> np.ndarray:
    """Bottom bar: frame counter + max joint angle difference at this frame."""
    img = img.copy()
    bar_y = H - 28
    cv2.rectangle(img, (0, bar_y), (W, H), (18, 18, 18), -1)
    txt_frame = f"frame {fi+1}/{N_FRAMES}"
    cv2.putText(img, txt_frame, (10, H - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA)
    # max joint diff across finger joints this frame
    max_diff_deg = np.degrees(joint_diff[fi])
    txt_diff = f"max finger joint diff: {max_diff_deg:.1f} deg"
    (tw, _), _ = cv2.getTextSize(txt_diff, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(img, txt_diff, (W - tw - 10, H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 100), 1, cv2.LINE_AA)
    return img


def add_legend(img: np.ndarray, is_left: bool) -> np.ndarray:
    img = img.copy()
    n = len(FINGERS) + 1
    lh = 22
    box_y = 80
    cv2.rectangle(img, (0, box_y), (160, box_y + n * lh + 10), (18, 18, 18), -1)
    for i, fdef in enumerate(FINGERS):
        y = box_y + 16 + i * lh
        b, gv, r = fdef["bgr"]
        cv2.circle(img, (16, y - 4), 7, (b, gv, r), -1)
        cv2.putText(img, fdef["name"], (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (210, 210, 210), 1, cv2.LINE_AA)
    # cross panel entry
    y = box_y + 16 + len(FINGERS) * lh
    cv2.circle(img, (16, y - 4), 5, (200, 200, 200), -1)
    other_label = "act-scene IK" if is_left else "standard IK"
    cv2.putText(img, other_label, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (160, 160, 160), 1, cv2.LINE_AA)
    return img


# ─── main ─────────────────────────────────────────────────────────────────────

def main(out: str):
    # ── load ──
    panel_data = []
    for p in PANELS:
        raw    = np.load(p["npz"])["qpos"]   # (T, nq)
        frames = [raw[i] for i in np.linspace(0, len(raw) - 1, N_FRAMES).astype(int)]
        model, data, renderer = make_model_data(p["scene"])
        site_ids = get_site_ids(model)
        panel_data.append({**p, "frames": frames, "model": model,
                           "data": data, "renderer": renderer, "site_ids": site_ids})

    # ── precompute fingertip world trajectories ──
    print("Precomputing fingertip trajectories...")
    for pd in panel_data:
        pd["traj"] = precompute_traj(pd["frames"], pd["model"], pd["data"], pd["site_ids"])
        print(f"  {pd['id']}: {pd['traj'].shape}")

    # ── per-frame max finger joint angle difference ──
    kin_qpos = np.array(panel_data[0]["frames"])   # (N, 33)
    act_qpos = np.array(panel_data[1]["frames"])   # (N, 32)
    joint_diff_per_frame = np.abs(kin_qpos[:, 6:26] - act_qpos[:, 6:26]).max(axis=1)  # (N,)

    # ── render ──
    print(f"Rendering {N_FRAMES} frames × {len(PANELS)} panels at {W}×{H}...")
    images = []
    for fi in range(N_FRAMES):
        if fi % 25 == 0:
            print(f"  frame {fi}/{N_FRAMES}")

        row = []
        for i, pd in enumerate(panel_data):
            other_traj = panel_data[1 - i]["traj"]

            qpos = pd["frames"][fi]
            pd["data"].qpos[:len(qpos)] = qpos
            mujoco.mj_forward(pd["model"], pd["data"])

            img = render_frame(pd["renderer"], pd["model"], pd["data"],
                               pd["traj"], other_traj, fi)
            img = add_header(img, pd["title"], pd["subtitle"], pd["note"])
            img = add_footer(img, fi, joint_diff_per_frame)
            img = add_legend(img, is_left=(i == 0))
            row.append(img)

        # divider between panels
        divider = np.full((H, 4, 3), 60, dtype=np.uint8)
        images.append(np.concatenate([row[0], divider, row[1]], axis=1))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, images, fps=FPS)
    W_total = W * 2 + 4
    print(f"\nSaved → {out}  ({N_FRAMES} frames, {FPS} fps, {W_total}×{H})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/compare_ik.mp4")
    args = parser.parse_args()
    main(args.out)
