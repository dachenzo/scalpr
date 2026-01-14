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
