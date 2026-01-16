from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class HandMetrics:
    total_frames: int
    detected_frames: int
    detection_rate: float
    valid_joint_percentage: float
    longest_missing_streak: int


@dataclass(slots=True)
class QualityThresholds:
    min_detection_rate: float = 0.90
    min_valid_joint_percentage: float = 0.90
    max_missing_streak: int = 15
    max_frame_count_mismatch: int = 0
    max_detected_frame_mismatch: int = 30


def _load_positions(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        times = data["times"] if "times" in data.files else np.array([], dtype=np.float64)
        if "raw_positions" in data.files:
            positions = data["raw_positions"]
        elif "positions" in data.files:
            positions = data["positions"]
        else:
            raise KeyError(f"No positions array found in {npz_path}")
    return times, positions


def _hand_metrics(positions: np.ndarray) -> HandMetrics:
    if positions.ndim != 3 or positions.shape[1:] != (21, 3):
        raise ValueError(f"Expected shape (T, 21, 3), got {positions.shape}")

    total_frames = int(positions.shape[0])
    if total_frames == 0:
        return HandMetrics(0, 0, 0.0, 0.0, 0)

    valid_joint_mask = np.all(np.isfinite(positions), axis=2)  # (T, 21)
    detected_frame_mask = np.any(valid_joint_mask, axis=1)  # (T,)

    detected_frames = int(detected_frame_mask.sum())
    detection_rate = float(detected_frames / total_frames)

    valid_joint_count = int(valid_joint_mask.sum())
    total_joint_count = int(total_frames * 21)
    valid_joint_percentage = float(valid_joint_count / total_joint_count) if total_joint_count > 0 else 0.0

    longest_missing_streak = 0
    current_streak = 0
    for has_detection in detected_frame_mask:
        if has_detection:
            current_streak = 0
        else:
            current_streak += 1
            if current_streak > longest_missing_streak:
                longest_missing_streak = current_streak

    return HandMetrics(
        total_frames=total_frames,
        detected_frames=detected_frames,
        detection_rate=detection_rate,
        valid_joint_percentage=valid_joint_percentage,
        longest_missing_streak=longest_missing_streak,
    )
