"""
Combine SPIDER IK wrist/object positions with wuji-retargeting finger angles.

Both SPIDER and wuji-retargeting share the same starting point:
  trajectory_ikrollout.npz (wrist world position + object position)

wuji-retargeting only replaces the finger DOFs:
  qpos[6:26]  (right fingers, 20 DOF)
  qpos[32:52] (left fingers, 20 DOF)

Usage:
    python src/wuji_from_ik.py \\
        --pkl       <anno_preview.pkl> \\
        --ik        <trajectory_ikrollout.npz> \\
        --config    configs/wuji_retarget_right.yaml \\
        --output    outputs/wuji_qpos_uncap.npy \\
        --start-frame 2255 --end-frame 2317
"""

import argparse
import os
import pickle

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANO_ROOT = "/home/roy/externalTool/data/OakInk2/asset/mano_v1_2"
SUBSAMPLE = 4


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
    rh_tsl   = torch.cat([data["raw_mano"][f]["rh__tsl"]         for f in frames])
    lh_pose  = torch.cat([data["raw_mano"][f]["lh__pose_coeffs"] for f in frames])
    lh_betas = torch.cat([data["raw_mano"][f]["lh__betas"]       for f in frames])
    lh_tsl   = torch.cat([data["raw_mano"][f]["lh__tsl"]         for f in frames])
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


def interp_trajectory(src, T_target):
    """Linear interpolate (T_src, nq) → (T_target, nq)."""
    T_src = src.shape[0]
    idx   = np.linspace(0, T_src - 1, T_target)
    i0    = idx.astype(int)
    i1    = np.minimum(i0 + 1, T_src - 1)
    alpha = (idx - i0)[:, None]
    return ((1 - alpha) * src[i0] + alpha * src[i1]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",         required=True,  help="anno_preview pkl")
    parser.add_argument("--ik",          required=True,  help="trajectory_ikrollout.npz")
    parser.add_argument("--config",      default=os.path.join(REPO_ROOT, "configs", "wuji_retarget_right.yaml"))
    parser.add_argument("--output",      required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame",   type=int, default=-1)
    args = parser.parse_args()
    args.config = os.path.abspath(args.config)
    args.output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. IK rollout: wrist + object positions in SPIDER scene coordinates
    ik_qpos = np.load(args.ik)["qpos"]   # (T_ik, 66)
    print(f"IK rollout: {ik_qpos.shape}  wrist_R={ik_qpos[0,:3].round(3)}  obj={ik_qpos[0,52:55].round(3)}", flush=True)

    # 2. MANO FK for wuji-retargeting finger angles
    print("Loading MANO FK...", flush=True)
    data, frames = load_anno(args.pkl, args.start_frame, args.end_frame)
    T = len(frames)
    print(f"Anno frames: {T}", flush=True)
    rh_joints, lh_joints = run_mano_fk(data, frames)

    # 3. Interpolate IK to match anno frame count
    ik_base = interp_trajectory(ik_qpos, T)

    # 4. Finger retargeting (only replaces DOF 6:26 and 32:52)
    print("Retargeting fingers...", flush=True)
    rh_fingers = retarget_fingers(rh_joints, args.config, "right")  # (T, 20)
    lh_fingers = retarget_fingers(lh_joints, args.config, "left")   # (T, 20)

    # 5. Assemble: IK base (wrist+objects) + wuji fingers
    full_qpos = ik_base.copy()
    full_qpos[:, 6:26]  = rh_fingers   # right fingers override
    full_qpos[:, 32:52] = lh_fingers   # left fingers override

    np.save(args.output, full_qpos)
    print(f"Saved: {args.output}  shape={full_qpos.shape}", flush=True)


if __name__ == "__main__":
    main()
