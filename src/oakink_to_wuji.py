"""
OakInk v2 → Wuji Hand qpos via wuji-retargeting (independent of SPIDER physics)

Data sources:
  --pkl       anno_preview pkl  → MANO FK → 21×3 joints → wuji-retargeting fingers
  --maniptrans maniptrans pkl   → wrist_pos/wrist_rot in scene coordinates
  --scene-obj-ids               → obj_transf → freejoint object positions

Output shape: (T, 52 + n_obj*7)
  [0:6]   right wrist 6DOF (tx,ty,tz,rx,ry,rz in SPIDER scene frame)
  [6:26]  right finger joints (20 DOF from wuji-retargeting)
  [26:32] left wrist
  [32:52] left finger joints
  [52:]   object freejoints (pos3 + quat_wxyz4 each)

Usage:
    python src/oakink_to_wuji.py \\
        --pkl         <anno_preview.pkl> \\
        --maniptrans  <task_bimanual.pkl> \\
        --config      configs/wuji_retarget_right.yaml \\
        --output      outputs/wuji_qpos.npy \\
        --start-frame 2255 --end-frame 2317 \\
        --scene-obj-ids O02@0206@00001 O02@0206@00002
"""

import argparse
import os
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANO_ROOT = "/home/roy/externalTool/data/OakInk2/asset/mano_v1_2"
SUBSAMPLE = 4

# OakInk → SPIDER scene coordinate transform (Y-up → Z-up)
R_GLOBAL = Rotation.from_euler("xyz", [np.pi / 2, 0, 0])


def load_anno(pkl_path, start_frame=0, end_frame=-1, subsample=SUBSAMPLE):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    frames = data["mocap_frame_id_list"][::subsample]
    end = end_frame if end_frame >= 0 else len(frames)
    return data, frames[start_frame:end]


def run_mano_fk(data, frames):
    from manotorch.manolayer import ManoLayer
    mano_r = ManoLayer(rot_mode="quat", side="right", mano_assets_root=MANO_ROOT, flat_hand_mean=False)
    mano_l = ManoLayer(rot_mode="quat", side="left",  mano_assets_root=MANO_ROOT, flat_hand_mean=False)
    rh_pose  = torch.cat([data["raw_mano"][f]["rh__pose_coeffs"] for f in frames])
    rh_betas = torch.cat([data["raw_mano"][f]["rh__betas"]       for f in frames])
    lh_pose  = torch.cat([data["raw_mano"][f]["lh__pose_coeffs"] for f in frames])
    lh_betas = torch.cat([data["raw_mano"][f]["lh__betas"]       for f in frames])
    lh_tsl   = torch.cat([data["raw_mano"][f]["lh__tsl"]         for f in frames])
    rh_tsl   = torch.cat([data["raw_mano"][f]["rh__tsl"]         for f in frames])
    with torch.no_grad():
        rh_out = mano_r(rh_pose, rh_betas)
        lh_out = mano_l(lh_pose, lh_betas)
    rh_joints = (rh_out.joints.numpy() + rh_tsl.numpy()[:, None, :])
    lh_joints = (lh_out.joints.numpy() + lh_tsl.numpy()[:, None, :])
    return rh_joints, lh_joints


def retarget_fingers(joints_T21x3, config_path, hand_side):
    from wuji_retargeting import Retargeter
    retargeter = Retargeter.from_yaml(config_path, hand_side=hand_side)
    T = joints_T21x3.shape[0]
    qpos = np.zeros((T, retargeter.num_joints), dtype=np.float32)
    for i, kp21 in enumerate(joints_T21x3):
        qpos[i] = retargeter.retarget(kp21, apply_filter=True)
    return qpos   # (T, 20)


def interp_wrist_from_maniptrans(maniptrans_pkl, T_target, side="right"):
    """
    Load wrist_pos/wrist_rot from maniptrans pkl, apply OakInk→SPIDER transform,
    interpolate to T_target frames, convert to 6DOF (tx,ty,tz,rx,ry,rz).

    wrist_rot in maniptrans is axis-angle (rotvec), same convention as oakink.py.
    Wrist offset: Rx(π/2) · Rz(π) — same as oakink.py.
    """
    with open(maniptrans_pkl, "rb") as f:
        mt = pickle.load(f)
    data = mt[side]

    wp = data["wrist_pos"].cpu().numpy()   # (T_src, 3) in OakInk world frame
    wr = data["wrist_rot"].cpu().numpy()   # (T_src, 3) axis-angle in OakInk frame

    r_wrist_offset = (
        Rotation.from_euler("xyz", [np.pi / 2, 0, 0])
        * Rotation.from_euler("xyz", [0, 0, np.pi])
    )
    T_src = len(wp)

    # Interpolation indices
    idx = np.linspace(0, T_src - 1, T_target)
    i0  = idx.astype(int)
    i1  = np.minimum(i0 + 1, T_src - 1)
    a   = (idx - i0)[:, None]

    wp_interp = (1 - a) * wp[i0] + a * wp[i1]   # (T_target, 3)
    wr_interp = (1 - a) * wr[i0] + a * wr[i1]   # (T_target, 3)

    wrist6 = np.zeros((T_target, 6), dtype=np.float32)
    for i in range(T_target):
        wrist6[i, :3] = R_GLOBAL.apply(wp_interp[i])
        r_total = R_GLOBAL * Rotation.from_rotvec(wr_interp[i]) * r_wrist_offset
        euler = r_total.as_euler("ZXY", degrees=False)
        wrist6[i, 3] =  euler[0]
        wrist6[i, 4] =  euler[1]
        wrist6[i, 5] = -euler[2]
    return wrist6


def obj_transf_to_freejoint(data, frames, obj_ids):
    """Extract object freejoint qpos (pos3 + wxyz4) with OakInk→SPIDER transform."""
    T = len(frames)
    obj_qpos = np.zeros((T, len(obj_ids) * 7), dtype=np.float32)
    for j, obj_id in enumerate(obj_ids):
        transf_dict = data["obj_transf"].get(obj_id, {})
        for i, fid in enumerate(frames):
            mat4 = transf_dict.get(fid)
            if mat4 is None:
                continue
            pos = R_GLOBAL.apply(mat4[:3, 3])
            rot = R_GLOBAL * Rotation.from_matrix(mat4[:3, :3])
            xyzw = rot.as_quat()
            wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            obj_qpos[i, j*7:j*7+3] = pos
            obj_qpos[i, j*7+3:j*7+7] = wxyz
    return obj_qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",          required=True, help="anno_preview pkl")
    parser.add_argument("--maniptrans",   required=True, help="task_bimanual.pkl (wrist positions in scene frame)")
    parser.add_argument("--config",       default=os.path.join(REPO_ROOT, "configs", "wuji_retarget_right.yaml"))
    parser.add_argument("--output",       required=True)
    parser.add_argument("--start-frame",  type=int, default=0)
    parser.add_argument("--end-frame",    type=int, default=-1)
    parser.add_argument("--scene-obj-ids", nargs="*", default=None)
    args = parser.parse_args()
    args.config = os.path.abspath(args.config)
    args.output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Anno preview → MANO FK → finger retargeting
    print(f"Loading anno_preview: {os.path.basename(args.pkl)}", flush=True)
    data, frames = load_anno(args.pkl, args.start_frame, args.end_frame)
    T = len(frames)
    print(f"Frames: {T}", flush=True)

    print("Running MANO FK...", flush=True)
    rh_joints, lh_joints = run_mano_fk(data, frames)

    print("Retargeting fingers → Wuji Hand...", flush=True)
    rh_fingers = retarget_fingers(rh_joints, args.config, "right")  # (T, 20)
    lh_fingers = retarget_fingers(lh_joints, args.config, "left")   # (T, 20)

    # 2. Maniptrans → wrist positions in SPIDER scene frame
    print(f"Loading wrist from maniptrans: {os.path.basename(args.maniptrans)}", flush=True)
    rh_wrist6 = interp_wrist_from_maniptrans(args.maniptrans, T, side="right")
    lh_wrist6 = interp_wrist_from_maniptrans(args.maniptrans, T, side="left")
    print(f"  right wrist frame 0: {rh_wrist6[0,:3].round(3)}", flush=True)
    print(f"  left  wrist frame 0: {lh_wrist6[0,:3].round(3)}", flush=True)

    # 3. Hand qpos: wrist + fingers
    hand_qpos = np.concatenate([rh_wrist6, rh_fingers, lh_wrist6, lh_fingers], axis=1)  # (T, 52)

    # 4. Object positions
    scene_obj_ids = args.scene_obj_ids or data["obj_list"]
    available = [o for o in scene_obj_ids if o in data["obj_transf"]]
    missing   = [o for o in scene_obj_ids if o not in data["obj_transf"]]
    if missing:
        print(f"Warning: objects not in pkl (will be zeros): {missing}", flush=True)
    print(f"Extracting object poses: {available}", flush=True)
    obj_qpos = obj_transf_to_freejoint(data, frames, scene_obj_ids)  # (T, n*7)
    print(f"  obj[0] frame 0: {obj_qpos[0,:3].round(3)}", flush=True)

    # 5. Full qpos
    full_qpos = np.concatenate([hand_qpos, obj_qpos], axis=1)
    np.save(args.output, full_qpos)
    print(f"Saved: {args.output}  shape={full_qpos.shape}", flush=True)


if __name__ == "__main__":
    main()
