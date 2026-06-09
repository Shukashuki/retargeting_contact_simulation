"""
Render MuJoCo scene + trajectory to mp4.

Uses the scene XML's "front" camera by default (same as SPIDER's render_trajectory.py).
Falls back to a free camera if "front" is not defined.

Usage:
    python src/render_scene.py \\
        --scene  <scene.xml> \\
        --traj   <trajectory_mjwp.npz | wuji_qpos.npy> \\
        --method spider | wuji \\
        --output outputs/render.mp4 \\
        [--width 640] [--height 480] [--fps 30]
"""

import argparse
import os
import sys

os.environ["MUJOCO_GL"] = "osmesa"   # WSL2 software renderer

import mujoco
import numpy as np
import imageio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from record_contact_hdf5 import load_spider_traj, load_wuji_traj


def render(scene_xml: str, traj: np.ndarray,
           output_path: str, width: int, height: int, fps: float,
           physics: bool = False) -> None:
    model   = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data = mujoco.MjData(model)
    nq      = model.nq
    nu      = model.nu
    T       = traj.shape[0]
    dt      = 1.0 / fps

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    camera = "front" if cam_id >= 0 else 0
    print(f"Using scene camera: {'front' if cam_id >= 0 else 'default'}", flush=True)

    if physics:
        sim_dt = model.opt.timestep
        nsteps = max(1, round(dt / sim_dt))
        print(f"  physics mode: sim_dt={sim_dt:.4f}s  nsteps/frame={nsteps}", flush=True)
        mj_data.qpos[:nq] = traj[0, :nq]
        mj_data.qvel[:]   = 0.0
        mujoco.mj_forward(model, mj_data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)

    print(f"Rendering {T} frames...", flush=True)
    for i in range(T):
        if physics:
            mj_data.ctrl[:nu] = traj[i, :nu]
            for _ in range(nsteps):
                mujoco.mj_step(model, mj_data)
        else:
            mj_data.qpos[:nq] = traj[i, :nq]
            mujoco.mj_forward(model, mj_data)

        renderer.update_scene(mj_data, camera=camera)
        writer.append_data(renderer.render())
        if i % 100 == 0:
            print(f"  {i}/{T}", flush=True)

    writer.close()
    print(f"Saved: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  required=True)
    parser.add_argument("--traj",   required=True)
    parser.add_argument("--method", required=True, choices=["spider", "wuji"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--width",  type=int,   default=640)
    parser.add_argument("--height", type=int,   default=480)
    parser.add_argument("--fps",    type=float, default=30.0)
    parser.add_argument("--physics", action="store_true",
                        help="Physics mode: hand follows PD control, objects respond freely")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    nq    = model.nq
    del model

    if args.method == "spider":
        traj = load_spider_traj(args.traj)
    else:
        traj = load_wuji_traj(args.traj, nq)

    print(f"Trajectory: {traj.shape}", flush=True)
    render(args.scene, traj, args.output, args.width, args.height, args.fps,
           physics=args.physics)


if __name__ == "__main__":
    main()
