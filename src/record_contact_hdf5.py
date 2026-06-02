"""
MuJoCo replay + HDF5 contact force recorder

Usage:
    python src/record_contact_hdf5.py \
        --scene  <scene.xml> \
        --traj   outputs/wuji_qpos.npy \
        --method wuji \
        --output outputs/recordings/wuji_uncap.h5
"""

import argparse
import os
import time

import h5py
import mujoco
import numpy as np

MAX_CONTACTS = 32


def load_traj(traj_path: str) -> np.ndarray:
    """Load qpos sequence from .npy or .npz (trajectory_mjwp_act.npz format)."""
    if traj_path.endswith(".npy"):
        return np.load(traj_path)   # (T, nq)
    # SPIDER npz: build qpos from wrist + finger arrays
    d = np.load(traj_path)
    if "qpos_wrist_right" in d:
        qw_r = d["qpos_wrist_right"]   # (T, 7)
        qf_r = d["qpos_finger_right"]  # (T, 5, 7)
        qw_l = d["qpos_wrist_left"]
        qf_l = d["qpos_finger_left"]
        T = qw_r.shape[0]
        return np.concatenate([
            qw_r, qf_r.reshape(T, -1),
            qw_l, qf_l.reshape(T, -1),
        ], axis=1)
    # fallback: try "qpos" key
    return d["qpos"]


def record(scene_xml: str, traj: np.ndarray, output_path: str,
           method: str, fps: float = 30.0):
    model  = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data = mujoco.MjData(model)
    T      = traj.shape[0]
    nq     = model.nq
    ns     = model.nsensordata

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        grp = h5.create_group("metadata")
        grp.attrs["method"] = method
        grp.attrs["scene"]  = scene_xml
        grp.attrs["fps"]    = fps
        grp.attrs["nframes"] = T

        h5.create_dataset("qpos",           shape=(T, nq),              dtype="f4")
        h5.create_dataset("contact_force",  shape=(T, MAX_CONTACTS, 6), dtype="f4")
        h5.create_dataset("ncon",           shape=(T,),                 dtype="i4")
        h5.create_dataset("sensor",         shape=(T, ns),              dtype="f4")
        h5.create_dataset("time",           shape=(T,),                 dtype="f8")

        dt = 1.0 / fps
        for i in range(T):
            mj_data.qpos[:nq] = traj[i, :nq]
            mujoco.mj_forward(model, mj_data)

            h5["qpos"][i]   = mj_data.qpos
            h5["time"][i]   = mj_data.time
            h5["sensor"][i] = mj_data.sensordata
            h5["ncon"][i]   = mj_data.ncon

            cf = np.zeros((MAX_CONTACTS, 6), dtype=np.float32)
            nc = min(mj_data.ncon, MAX_CONTACTS)
            for c in range(nc):
                contact = mj_data.contact[c]
                # rotate contact frame force into world frame (3D → 6D wrench)
                frame  = contact.frame.reshape(3, 3)
                force3 = contact.dist * frame[:, 0]  # normal force
                cf[c, :3] = force3
            h5["contact_force"][i] = cf

            mj_data.time += dt

        print(f"✓ 錄製完成: {output_path}  ({T} 幀)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  required=True)
    parser.add_argument("--traj",   required=True)
    parser.add_argument("--method", default="wuji", choices=["wuji", "spider"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps",    type=float, default=30.0)
    args = parser.parse_args()

    print(f"載入軌跡: {args.traj}", flush=True)
    traj = load_traj(args.traj)
    print(f"幀數={traj.shape[0]}, DOF={traj.shape[1]}", flush=True)

    record(args.scene, traj, args.output, method=args.method, fps=args.fps)


if __name__ == "__main__":
    main()
