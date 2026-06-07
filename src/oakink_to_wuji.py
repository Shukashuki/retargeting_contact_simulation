"""
OakInk v2 anno_preview pkl → Wuji Hand qpos sequence (via wuji-retargeting)

Output shape: (T, 52) — matches SPIDER 6DOF-wrist bimanual model
  [0:6]   right wrist (tx,ty,tz from MANO tsl + rx,ry,rz from MANO global rot)
  [6:26]  right finger joints (20 DOF from wuji-retargeting)
  [26:32] left wrist
  [32:52] left finger joints

Usage:
    python src/oakink_to_wuji.py --pkl <path_to_anno.pkl> \
        --config configs/wuji_retarget_right.yaml \
        --output outputs/wuji_qpos.npy
"""

import argparse
import os
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANO_ROOT = "/home/roy/externalTool/data/OakInk2/asset/mano_v1_2"

FPS       = 30
SUBSAMPLE = 4   # 120 Hz mocap → ~30 Hz
# manotorch reorder: wrist=0, thumb_tip=4, index_tip=8, middle_tip=12, ring_tip=16, pinky_tip=20
JOINT_IDX_ALL = list(range(21))   # all 21 joints for wuji-retargeting


def load_anno(pkl_path: str, subsample: int = SUBSAMPLE):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    frames   = data["mocap_frame_id_list"][::subsample]
    obj_list = data["obj_list"]
    return data, frames, obj_list


def run_mano_fk(data, frames):
    """Run MANO forward kinematics, return (T, 21, 3) world-space joints for each hand."""
    from manotorch.manolayer import ManoLayer

    mano_r = ManoLayer(rot_mode="quat", side="right", mano_assets_root=MANO_ROOT, flat_hand_mean=False)
    mano_l = ManoLayer(rot_mode="quat", side="left",  mano_assets_root=MANO_ROOT, flat_hand_mean=False)

    rh_pose  = torch.cat([data["raw_mano"][f]["rh__pose_coeffs"] for f in frames])
    rh_betas = torch.cat([data["raw_mano"][f]["rh__betas"]       for f in frames])
    rh_tsl   = torch.cat([data["raw_mano"][f]["rh__tsl"]         for f in frames])
    lh_pose  = torch.cat([data["raw_mano"][f]["lh__pose_coeffs"] for f in frames])
    lh_betas = torch.cat([data["raw_mano"][f]["lh__betas"]       for f in frames])
    lh_tsl   = torch.cat([data["raw_mano"][f]["lh__tsl"]         for f in frames])

    with torch.no_grad():
        rh_out = mano_r(rh_pose, rh_betas)
        lh_out = mano_l(lh_pose, lh_betas)

    # World-space joints: (T, 21, 3)
    rh_joints = (rh_out.joints.numpy() + rh_tsl.numpy()[:, None, :])
    lh_joints = (lh_out.joints.numpy() + lh_tsl.numpy()[:, None, :])
    return rh_joints, lh_joints


def wrist_pose_to_6dof(tsl: np.ndarray, pose_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert MANO wrist world pose to SPIDER 6DOF wrist joint values,
    applying the OakInk→SPIDER global coordinate transform first:
        r_global = Rx(π/2)   (Y-up OakInk → Z-up SPIDER)

    Wrist offset matches oakink.py:
        r_wrist_offset = Rx(π/2) · Rz(π)
    """
    r_global = Rotation.from_euler("xyz", [np.pi / 2, 0, 0])
    r_wrist_offset = (
        Rotation.from_euler("xyz", [np.pi / 2, 0, 0])
        * Rotation.from_euler("xyz", [0, 0, np.pi])
    )
    T = tsl.shape[0]
    wrist_6dof = np.zeros((T, 6), dtype=np.float32)

    for i in range(T):
        # Position: apply global rotation
        wrist_6dof[i, :3] = r_global.apply(tsl[i])

        # Rotation: r_global * MANO_global_rot * r_wrist_offset → ZXY Euler for chain
        global_quat_xyzw = pose_coeffs[i, 0, :]       # [x,y,z,w]
        r_mano = Rotation.from_quat(global_quat_xyzw)
        r_total = r_global * r_mano * r_wrist_offset
        euler_zxy = r_total.as_euler('ZXY', degrees=False)  # [α_z, β_x, γ_y]
        wrist_6dof[i, 3] = euler_zxy[0]               # rx_joint (axis Z)
        wrist_6dof[i, 4] = euler_zxy[1]               # ry_joint (axis X)
        wrist_6dof[i, 5] = -euler_zxy[2]              # rz_joint (axis -Y, negated)

    return wrist_6dof


def retarget_sequence(joints_T21x3: np.ndarray, config_path: str, hand_side: str) -> np.ndarray:
    """
    joints_T21x3 : (T, 21, 3) world-space MANO joints
    Returns      : (T, 20) Wuji Hand qpos
    """
    from wuji_retargeting import Retargeter

    retargeter = Retargeter.from_yaml(config_path, hand_side=hand_side)
    T = joints_T21x3.shape[0]
    qpos_seq = np.zeros((T, retargeter.num_joints), dtype=np.float32)

    for i, kp21 in enumerate(joints_T21x3):
        qpos_seq[i] = retargeter.retarget(kp21, apply_filter=True)

    return qpos_seq


def obj_transf_to_freejoint(data: dict, frames: list, obj_ids: list) -> np.ndarray:
    """
    Extract object world transforms from obj_transf → freejoint qpos (pos3 + quat_wxyz4).
    Applies the same OakInk→SPIDER global rotation as oakink.py: Rx(π/2).
    """
    r_global = Rotation.from_euler("xyz", [np.pi / 2, 0, 0])
    T = len(frames)
    obj_qpos = np.zeros((T, len(obj_ids) * 7), dtype=np.float32)
    for j, obj_id in enumerate(obj_ids):
        transf_dict = data["obj_transf"].get(obj_id, {})
        for i, fid in enumerate(frames):
            mat4 = transf_dict.get(fid)
            if mat4 is None:
                continue
            pos = r_global.apply(mat4[:3, 3])
            rot = r_global * Rotation.from_matrix(mat4[:3, :3])
            xyzw = rot.as_quat()                      # scipy: [x,y,z,w]
            wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            obj_qpos[i, j*7:j*7+3] = pos
            obj_qpos[i, j*7+3:j*7+7] = wxyz
    return obj_qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",    required=True, help="anno_preview pkl path")
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "wuji_retarget_right.yaml"))
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "outputs", "wuji_qpos.npy"))
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE)
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start mocap frame index (before subsampling)")
    parser.add_argument("--end-frame",   type=int, default=-1,
                        help="End mocap frame index (before subsampling), -1 = all")
    parser.add_argument("--scene-obj-ids", nargs="*", default=None,
                        help="Object IDs to include in qpos (in scene order). "
                             "If omitted, all obj_list objects are used.")
    args = parser.parse_args()
    args.config = os.path.abspath(args.config)
    args.output = os.path.abspath(args.output)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading pkl: {args.pkl}", flush=True)
    data, frames, obj_list = load_anno(args.pkl, args.subsample)
    # Crop to requested frame window
    end = args.end_frame if args.end_frame >= 0 else len(frames)
    frames = frames[args.start_frame:end]
    print(f"Frames: {len(frames)}  Objects: {obj_list}", flush=True)

    print("Running MANO FK...", flush=True)
    rh_joints, lh_joints = run_mano_fk(data, frames)

    rh_tsl  = np.stack([data["raw_mano"][f]["rh__tsl"][0].numpy()         for f in frames])
    lh_tsl  = np.stack([data["raw_mano"][f]["lh__tsl"][0].numpy()         for f in frames])
    rh_pose = np.stack([data["raw_mano"][f]["rh__pose_coeffs"][0].numpy() for f in frames])
    lh_pose = np.stack([data["raw_mano"][f]["lh__pose_coeffs"][0].numpy() for f in frames])

    rh_wrist6 = wrist_pose_to_6dof(rh_tsl, rh_pose)
    lh_wrist6 = wrist_pose_to_6dof(lh_tsl, lh_pose)

    print("Retargeting fingers → Wuji Hand...", flush=True)
    rh_fingers = retarget_sequence(rh_joints, args.config, hand_side="right")
    lh_fingers = retarget_sequence(lh_joints, args.config, hand_side="left")

    # Hand qpos: (T, 52)
    hand_qpos = np.concatenate([rh_wrist6, rh_fingers, lh_wrist6, lh_fingers], axis=1)

    # Object qpos: (T, n_obj * 7)
    scene_obj_ids = args.scene_obj_ids if args.scene_obj_ids else obj_list
    available = [o for o in scene_obj_ids if o in data["obj_transf"]]
    missing   = [o for o in scene_obj_ids if o not in data["obj_transf"]]
    if missing:
        print(f"Warning: objects not in pkl, will be zero: {missing}", flush=True)
    print(f"Extracting object poses for: {available}", flush=True)
    obj_qpos = obj_transf_to_freejoint(data, frames, scene_obj_ids)  # (T, n*7)

    # Full qpos: hand (52) + objects (n*7)
    full_qpos = np.concatenate([hand_qpos, obj_qpos], axis=1)
    np.save(args.output, full_qpos)
    print(f"Saved: {args.output}  shape={full_qpos.shape}", flush=True)
    print(f"  right wrist: {rh_wrist6[0,:3].round(3)}", flush=True)
    print(f"  obj[0] pos:  {obj_qpos[0,:3].round(3)}", flush=True)


if __name__ == "__main__":
    main()
