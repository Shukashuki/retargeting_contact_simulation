"""Render sim vs ref comparison for Mode A, B, C.

Layout (3 rows × 2 panels):
  Row 0: Mode A  [IK ref | sim]
  Row 1: Mode B  [IK ref | sim]
  Row 2: Mode C  [IK ref | sim]

Usage:
  python src/render_modes.py --out outputs/modes_comparison.mp4
"""

from __future__ import annotations
import argparse, os
import cv2, imageio, mujoco
import numpy as np

SPIDER  = os.path.join(os.path.dirname(__file__), "../third_party/spider")
RUN_DIR = os.path.join(SPIDER, "example_datasets/processed/oakinkv2/wuji_hand/right/unscrew_bottle/0")
SCENE     = os.path.join(RUN_DIR, "../scene.xml")
SCENE_ACT = os.path.join(RUN_DIR, "../scene_act.xml")

MODES = [
    {
        "label": "Mode A  qpos tracking",
        "ref":  {"scene": SCENE,     "npz": os.path.join(RUN_DIR, "trajectory_kinematic.npz"),       "flat": True},
        "sim":  {"scene": SCENE,     "npz": os.path.join(RUN_DIR, "trajectory_mjwp_no_guidance.npz"), "flat": False},
    },
    {
        "label": "Mode B  contact guidance",
        "ref":  {"scene": SCENE_ACT, "npz": os.path.join(RUN_DIR, "trajectory_kinematic_act.npz"),   "flat": True},
        "sim":  {"scene": SCENE_ACT, "npz": os.path.join(RUN_DIR, "trajectory_mjwp_act.npz"),         "flat": False},
    },
    {
        "label": "Mode C  geometric reward",
        "ref":  {"scene": SCENE,     "npz": os.path.join(RUN_DIR, "trajectory_kinematic.npz"),       "flat": True},
        "sim":  {"scene": SCENE,     "npz": os.path.join(RUN_DIR, "trajectory_mjwp_cap_contact.npz"), "flat": False},
    },
]

W, H         = 480, 480
FPS          = 25
N_FRAMES     = 196
RENDER_EVERY = 2

CAP_LOOKAT = np.array([0.036, -0.173, -0.041], dtype=np.float64)


def make_model_data(scene_path):
    m = mujoco.MjModel.from_xml_path(scene_path)
    m.vis.global_.offwidth  = W
    m.vis.global_.offheight = H
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=H, width=W)
    return m, d, r


def extract_frames(npz_path, flat, n):
    raw = np.load(npz_path)["qpos"]
    if flat:
        idxs = np.linspace(0, len(raw) - 1, n).astype(int)
        return [raw[i] for i in idxs]
    n_ticks, ctrl_steps, _ = raw.shape
    frames = []
    for tick in range(n_ticks):
        for step in range(ctrl_steps):
            if step % RENDER_EVERY == 0:
                frames.append(raw[tick, step])
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


def make_cam():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 0.35
    cam.azimuth   = -45
    cam.elevation = -20
    cam.lookat[:] = CAP_LOOKAT
    return cam


def render_one(renderer, model, data, qpos):
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    renderer.update_scene(data, camera=make_cam(), scene_option=opt)
    return renderer.render().copy()


def add_label(img, top_line, bot_line, is_ref):
    img = img.copy()
    tag   = "REF (IK)" if is_ref else "SIM"
    color = (180, 220, 180) if is_ref else (220, 180, 180)   # green-ish / red-ish
    cv2.rectangle(img, (0, 0), (W, 58), (20, 20, 20), -1)
    cv2.putText(img, top_line, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(img, bot_line, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color,          1, cv2.LINE_AA)
    # tag top-right
    (tw,_),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (W-tw-14, 0), (W, 30), (40,40,40), -1)
    cv2.putText(img, tag, (W-tw-7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img


def add_frame_counter(img, fi):
    img = img.copy()
    txt = f"{fi+1}/{N_FRAMES}"
    cv2.putText(img, txt, (8, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,100,100), 1, cv2.LINE_AA)
    return img


def main(out):
    # load everything up front
    rows = []
    for mode in MODES:
        row = {}
        for side in ("ref", "sim"):
            cfg = mode[side]
            m, d, r = make_model_data(cfg["scene"])
            frames   = extract_frames(cfg["npz"], cfg["flat"], N_FRAMES)
            row[side] = {"model": m, "data": d, "renderer": r, "frames": frames}
        row["label"] = mode["label"]
        rows.append(row)

    print(f"Rendering {N_FRAMES} frames  ({len(MODES)} modes × 2 panels)...")
    images = []
    for fi in range(N_FRAMES):
        if fi % 25 == 0:
            print(f"  frame {fi}/{N_FRAMES}")

        video_rows = []
        for row in rows:
            mode_label = row["label"]
            panels = []
            for side, is_ref in [("ref", True), ("sim", False)]:
                p  = row[side]
                qp = p["frames"][fi]
                img = render_one(p["renderer"], p["model"], p["data"], qp)
                bot = "IK reference" if is_ref else "physics optimization"
                img = add_label(img, mode_label, bot, is_ref)
                img = add_frame_counter(img, fi)
                panels.append(img)
            # thin divider between ref and sim
            div = np.full((H, 3, 3), 80, dtype=np.uint8)
            video_rows.append(np.concatenate([panels[0], div, panels[1]], axis=1))

        # thin horizontal divider between modes
        hdiv = np.full((3, W*2+3, 3), 60, dtype=np.uint8)
        frame = np.concatenate([video_rows[0], hdiv, video_rows[1], hdiv, video_rows[2]], axis=0)
        images.append(frame)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, images, fps=FPS)
    print(f"\nSaved → {out}  ({N_FRAMES} frames, {FPS}fps, {W*2+3}×{H*3+6})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/modes_comparison.mp4")
    args = parser.parse_args()
    main(args.out)
