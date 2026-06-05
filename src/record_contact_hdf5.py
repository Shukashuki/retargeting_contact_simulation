"""
MuJoCo replay + HDF5 contact force recorder.

Supports two trajectory formats:
  - SPIDER mjwp npz: qpos (n_ticks, ctrl_steps, nq)
  - wuji-retargeting npy: qpos (T, 40) — right 20 + left 20 DOF (no wrist DOF)

Usage:
    python src/record_contact_hdf5.py \\
        --scene  <scene.xml> \\
        --traj   <trajectory_mjwp.npz | wuji_qpos.npy> \\
        --method spider | wuji \\
        --output outputs/recordings/<name>.h5
"""

import argparse
import os

import h5py
import mujoco
import numpy as np

MAX_CONTACTS = 64


def load_spider_traj(npz_path: str) -> np.ndarray:
    """Load SPIDER mjwp trajectory. Returns (T, nq) qpos array."""
    d = np.load(npz_path)
    qpos = d["qpos"]          # (n_ticks, ctrl_steps, nq)
    n_ticks, ctrl_steps, nq = qpos.shape
    return qpos.reshape(n_ticks * ctrl_steps, nq)


def load_wuji_traj(npy_path: str, model_nq: int) -> np.ndarray:
    """
    Load wuji-retargeting output and pad to model nq.

    Supports two formats:
      (T, 52): [wrist6_R | fingers20_R | wrist6_L | fingers20_L]  ← new format with wrist
      (T, 40): [fingers20_R | fingers20_L]                         ← legacy finger-only
    """
    raw = np.load(npy_path)
    T, cols = raw.shape
    qpos = np.zeros((T, model_nq), dtype=np.float64)

    if cols == 52 and model_nq >= 52:
        # New format: wrist + fingers already in SPIDER DOF order
        qpos[:, :52] = raw
    elif cols == 40 and model_nq >= 52:
        # Legacy: finger-only, wrist stays at 0 (hands at origin)
        qpos[:, 6:26]  = raw[:, :20]   # right fingers
        qpos[:, 32:52] = raw[:, 20:]   # left fingers
        print("Warning: wrist position not included — hands will be at origin.", flush=True)
    else:
        n = min(cols, model_nq)
        qpos[:, :n] = raw[:, :n]
    return qpos


def record(scene_xml: str, traj: np.ndarray, output_path: str,
           method: str, fps: float = 30.0) -> None:
    model   = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data = mujoco.MjData(model)
    T   = traj.shape[0]
    nq  = model.nq
    ns  = model.nsensordata

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["method"]  = method
        meta.attrs["scene"]   = scene_xml
        meta.attrs["fps"]     = fps
        meta.attrs["nframes"] = T
        meta.attrs["nq"]      = nq

        ds_qpos = h5.create_dataset("qpos",          shape=(T, nq),              dtype="f4")
        ds_cf   = h5.create_dataset("contact_force",  shape=(T, MAX_CONTACTS, 6), dtype="f4")
        ds_ncon = h5.create_dataset("ncon",           shape=(T,),                 dtype="i4")
        ds_sens = h5.create_dataset("sensor",         shape=(T, ns),              dtype="f4")
        ds_time = h5.create_dataset("time",           shape=(T,),                 dtype="f8")

        dt = 1.0 / fps
        for i in range(T):
            frame_qpos = traj[i, :nq]
            mj_data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, mj_data)

            ds_qpos[i] = mj_data.qpos.astype(np.float32)
            ds_time[i] = i * dt
            ds_sens[i] = mj_data.sensordata.astype(np.float32)
            ds_ncon[i] = mj_data.ncon

            cf = np.zeros((MAX_CONTACTS, 6), dtype=np.float32)
            nc = min(mj_data.ncon, MAX_CONTACTS)
            for c in range(nc):
                contact = mj_data.contact[c]
                # Transform contact frame force into world frame
                force_local = np.zeros(6)
                mujoco.mj_contactForce(model, mj_data, c, force_local)
                frame = contact.frame.reshape(3, 3)
                cf[c, :3] = frame @ force_local[:3]   # force
                cf[c, 3:] = frame @ force_local[3:]   # torque
            ds_cf[i] = cf

            mj_data.time += dt

        print(f"Saved {T} frames → {output_path}", flush=True)
        print(f"  nq={nq}  ns={ns}  max_contacts={MAX_CONTACTS}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  required=True, help="MuJoCo scene XML")
    parser.add_argument("--traj",   required=True, help=".npz (SPIDER) or .npy (wuji)")
    parser.add_argument("--method", required=True, choices=["spider", "wuji"])
    parser.add_argument("--output", required=True, help="Output .h5 path")
    parser.add_argument("--fps",    type=float, default=30.0)
    args = parser.parse_args()

    # Resolve model nq from scene
    model = mujoco.MjModel.from_xml_path(args.scene)
    nq    = model.nq
    del model

    print(f"Loading trajectory ({args.method}): {args.traj}", flush=True)
    if args.method == "spider":
        traj = load_spider_traj(args.traj)
    else:
        traj = load_wuji_traj(args.traj, nq)

    print(f"Frames: {traj.shape[0]}  model nq: {nq}", flush=True)
    record(args.scene, traj, args.output, method=args.method, fps=args.fps)


if __name__ == "__main__":
    main()
