"""
觸覺熱圖可視化：讀 HDF5 sensor 資料 → 逐幀 heatmap → mp4

Usage:
    python src/viz_heatmap.py \
        --h5     outputs/recordings/wuji_uncap.h5 \
        --output outputs/heatmap.mp4 \
        --fps    30
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio


FINGER_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def sensor_to_heatmap_frame(sensor_t: np.ndarray, vmax: float = 10.0) -> np.ndarray:
    """
    sensor_t: (10,) — right hand fingers [0:5] + left hand fingers [5:10]
    Returns: RGB numpy array (H, W, 3) uint8
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=80)
    titles = ["Right Hand", "Left Hand"]
    cmap = plt.get_cmap("YlOrRd")

    for side_idx, ax in enumerate(axes):
        vals = sensor_t[side_idx * 5 : side_idx * 5 + 5].reshape(1, 5)  # (1, 5)
        im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(FINGER_LABELS, fontsize=7)
        ax.set_yticks([])
        ax.set_title(titles[side_idx], fontsize=9)

    fig.tight_layout(pad=1.0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",     required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps",    type=float, default=30.0)
    parser.add_argument("--vmax",   type=float, default=10.0, help="sensor saturation value")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.h5, "r") as h5:
        sensor = h5["sensor"][:]   # (T, ns)

    ns = sensor.shape[1]
    if ns < 10:
        print(f"⚠ sensor 維度={ns} < 10，將用 0 補齊", flush=True)
        pad = np.zeros((sensor.shape[0], 10 - ns), dtype=np.float32)
        sensor = np.concatenate([sensor, pad], axis=1)

    T = sensor.shape[0]
    print(f"生成 {T} 幀熱圖...", flush=True)

    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8)
    for i in range(T):
        frame = sensor_to_heatmap_frame(sensor[i, :10], vmax=args.vmax)
        writer.append_data(frame)
        if i % 100 == 0:
            print(f"  {i}/{T}", flush=True)
    writer.close()
    print(f"✓ 熱圖影片: {args.output}", flush=True)


if __name__ == "__main__":
    main()
