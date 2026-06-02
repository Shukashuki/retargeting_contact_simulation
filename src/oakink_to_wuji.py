"""
OakInk v2 anno_preview pkl → Wuji Hand qpos sequence (via wuji-retargeting)

Usage:
    python src/oakink_to_wuji.py --pkl <path_to_anno.pkl> \
        --config configs/wuji_retarget.yaml \
        --output outputs/wuji_qpos.npy
"""

import argparse
import os
import pickle

import numpy as np
import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",    required=True, help="anno_preview pkl path")
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "wuji_retarget_right.yaml"))
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "outputs", "wuji_qpos.npy"))
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE)
    args = parser.parse_args()
    # resolve to absolute paths immediately (before any os.chdir that might happen)
    args.config = os.path.abspath(args.config)
    args.output = os.path.abspath(args.output)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"載入 pkl: {args.pkl}", flush=True)
    data, frames, obj_list = load_anno(args.pkl, args.subsample)
    print(f"幀數: {len(frames)}, 物件: {obj_list}", flush=True)

    print("執行 MANO FK...", flush=True)
    rh_joints, lh_joints = run_mano_fk(data, frames)   # (T, 21, 3) each

    print("Retargeting → Wuji Hand...", flush=True)
    rh_qpos = retarget_sequence(rh_joints, args.config, hand_side="right")  # (T, 20)
    lh_qpos = retarget_sequence(lh_joints, args.config, hand_side="left")   # (T, 20)

    qpos_bimanual = np.concatenate([rh_qpos, lh_qpos], axis=1)   # (T, 40)
    np.save(args.output, qpos_bimanual)
    print(f"儲存: {args.output}  shape={qpos_bimanual.shape}", flush=True)


if __name__ == "__main__":
    main()
