from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
            continue
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


def _evaluate_warnings(
    left: HandMetrics,
    right: HandMetrics,
    frame_count_mismatch: int,
    detected_frame_mismatch: int,
    thresholds: QualityThresholds,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []

    for side, metrics in (("left", left), ("right", right)):
        if metrics.detection_rate < thresholds.min_detection_rate:
            warnings.append(
                {
                    "metric": f"{side}.hand_detection_rate",
                    "value": float(metrics.detection_rate),
                    "threshold": float(thresholds.min_detection_rate),
                    "kind": "below_min",
                }
            )
        if metrics.valid_joint_percentage < thresholds.min_valid_joint_percentage:
            warnings.append(
                {
                    "metric": f"{side}.valid_joint_percentage",
                    "value": float(metrics.valid_joint_percentage),
                    "threshold": float(thresholds.min_valid_joint_percentage),
                    "kind": "below_min",
                }
            )
        if metrics.longest_missing_streak > thresholds.max_missing_streak:
            warnings.append(
                {
                    "metric": f"{side}.longest_missing_streak",
                    "value": int(metrics.longest_missing_streak),
                    "threshold": int(thresholds.max_missing_streak),
                    "kind": "above_max",
                }
            )

    if frame_count_mismatch > thresholds.max_frame_count_mismatch:
        warnings.append(
            {
                "metric": "summary.left_right_frame_count_mismatch",
                "value": int(frame_count_mismatch),
                "threshold": int(thresholds.max_frame_count_mismatch),
                "kind": "above_max",
            }
        )

    if detected_frame_mismatch > thresholds.max_detected_frame_mismatch:
        warnings.append(
            {
                "metric": "summary.left_right_detected_frame_count_mismatch",
                "value": int(detected_frame_mismatch),
                "threshold": int(thresholds.max_detected_frame_mismatch),
                "kind": "above_max",
            }
        )

    return warnings


def generate_run_report(
    left_npz_path: Path,
    right_npz_path: Path,
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = QualityThresholds()

    left_times, left_positions = _load_positions(left_npz_path)
    right_times, right_positions = _load_positions(right_npz_path)

    left = _hand_metrics(left_positions)
    right = _hand_metrics(right_positions)

    frame_count_mismatch = abs(left.total_frames - right.total_frames)
    detected_frame_mismatch = abs(left.detected_frames - right.detected_frames)

    warnings = _evaluate_warnings(
        left=left,
        right=right,
        frame_count_mismatch=frame_count_mismatch,
        detected_frame_mismatch=detected_frame_mismatch,
        thresholds=thresholds,
    )

    total_frames_read = int(max(left.total_frames, right.total_frames, len(left_times), len(right_times)))

    report: dict[str, Any] = {
        "summary": {
            "total_frames_read": total_frames_read,
            "left_right_frame_count_mismatch": int(frame_count_mismatch),
            "left_right_detected_frame_count_mismatch": int(detected_frame_mismatch),
            "status": "warn" if warnings else "ok",
        },
        "left": {
            "detected_frames": left.detected_frames,
            "hand_detection_rate": left.detection_rate,
            "valid_joint_percentage": left.valid_joint_percentage,
            "longest_missing_streak": left.longest_missing_streak,
        },
        "right": {
            "detected_frames": right.detected_frames,
            "hand_detection_rate": right.detection_rate,
            "valid_joint_percentage": right.valid_joint_percentage,
            "longest_missing_streak": right.longest_missing_streak,
        },
        "thresholds": {
            "min_detection_rate": float(thresholds.min_detection_rate),
            "min_valid_joint_percentage": float(thresholds.min_valid_joint_percentage),
            "max_missing_streak": int(thresholds.max_missing_streak),
            "max_frame_count_mismatch": int(thresholds.max_frame_count_mismatch),
            "max_detected_frame_mismatch": int(thresholds.max_detected_frame_mismatch),
        },
        "warnings": warnings,
    }

    return report
