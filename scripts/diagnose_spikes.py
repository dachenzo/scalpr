from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _compute_speed(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    if positions.shape[0] < 2:
        return np.empty((0, positions.shape[1]), dtype=np.float64)

    dt = np.diff(times)
    delta = np.diff(positions, axis=0)
    speed = np.linalg.norm(delta, axis=2).astype(np.float64)

    valid_dt = np.isfinite(dt) & (dt > 1e-6)
    speed[~valid_dt, :] = np.nan
    speed[valid_dt, :] /= dt[valid_dt, np.newaxis]
    return speed


def _print_stats(name: str, positions: np.ndarray, times: np.ndarray, vmax: float) -> None:
    speed = _compute_speed(positions, times)
    if speed.size == 0:
        print(f"[{name}] No speed samples")
        return

    finite = np.isfinite(speed)
    spikes = finite & (speed > vmax)

    print(f"[{name}] samples={int(finite.sum())} spikes>{vmax:.2f}m/s={int(spikes.sum())}")
    if not np.any(spikes):
        return

    per_joint = spikes.sum(axis=0)
    top_joint_ids = np.argsort(per_joint)[::-1]
    print(f"[{name}] top spike joints:")
    shown = 0
    for joint_idx in top_joint_ids:
        count = int(per_joint[joint_idx])
        if count <= 0:
            continue
        print(f"  - joint {joint_idx:02d}: {count}")
        shown += 1
        if shown >= 8:
            break

    events = np.argwhere(spikes)
    print(f"[{name}] first spike events (frame_from->frame_to, joint, speed):")
    for edge_idx, joint_idx in events[:20]:
        value = speed[edge_idx, joint_idx]
        print(f"  - {edge_idx:05d}->{edge_idx+1:05d}, j{joint_idx:02d}, {value:.3f} m/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose non-human 3D hand joint spikes from NPZ outputs.")
    parser.add_argument("--npz", required=True, help="Path to hand npz output file.")
    parser.add_argument("--vmax", type=float, default=4.0, help="Speed threshold in m/s for spike events.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        times = data["times"]
        raw = data["raw_positions"] if "raw_positions" in data.files else data["positions"]
        despiked = data["despiked_positions"] if "despiked_positions" in data.files else None
        pre_post_despike_smoothed = (
            data["pre_post_despike_smoothed_positions"]
            if "pre_post_despike_smoothed_positions" in data.files
            else None
        )
        smoothed = data["smoothed_positions"] if "smoothed_positions" in data.files else data["positions"]
        spike_mask = data["spike_mask"] if "spike_mask" in data.files else None
        post_smoothing_spike_mask = (
            data["post_smoothing_spike_mask"] if "post_smoothing_spike_mask" in data.files else None
        )
        final_guard_spike_mask = data["final_guard_spike_mask"] if "final_guard_spike_mask" in data.files else None

    print(f"File: {npz_path}")
    print(f"Frames: {raw.shape[0]}")

    _print_stats("raw", raw, times, args.vmax)
    if despiked is not None:
        _print_stats("despiked", despiked, times, args.vmax)
    if pre_post_despike_smoothed is not None:
        _print_stats("smoothed_pre_post_despike", pre_post_despike_smoothed, times, args.vmax)
    _print_stats("smoothed", smoothed, times, args.vmax)

    if spike_mask is not None:
        print(f"[mask] flagged joints: {int(np.sum(spike_mask))}")
    if post_smoothing_spike_mask is not None:
        print(f"[post_smoothing_mask] flagged joints: {int(np.sum(post_smoothing_spike_mask))}")
    if final_guard_spike_mask is not None:
        print(f"[final_guard_mask] flagged joints: {int(np.sum(final_guard_spike_mask))}")


if __name__ == "__main__":
    main()
