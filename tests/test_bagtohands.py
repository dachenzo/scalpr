from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bagtohands import BagToHands


@dataclass
class FakeLandmark:
    x: float
    y: float


class FakeHandLandmarks:
    def __init__(self, landmarks: list[FakeLandmark]) -> None:
        self.landmark = landmarks


class BagToHandsUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = BagToHands.__new__(BagToHands)
        self.processor.depth_scale = 0.001
        self.processor.depth_min_meters = 0.08
        self.processor.depth_max_meters = 2.5

    def test_get_depth_at_returns_median_of_valid_patch(self) -> None:
        depth = np.array(
            [
                [0, 1000, 0],
                [900, 1100, 1300],
                [0, 1200, 0],
            ],
            dtype=np.uint16,
        )

        value = BagToHands._get_depth_at(self.processor, depth, u=1, v=1, window=1)

        self.assertAlmostEqual(value, 1.1, places=6)

    def test_get_depth_at_returns_zero_when_patch_has_no_valid_depth(self) -> None:
        depth = np.zeros((5, 5), dtype=np.uint16)

        value = BagToHands._get_depth_at(self.processor, depth, u=2, v=2, window=1)

        self.assertTrue(np.isnan(value))

    def test_landmarks_to_3d_projects_valid_landmark_and_preserves_nans(self) -> None:
        depth = np.full((10, 10), 1000, dtype=np.uint16)

        landmarks = [FakeLandmark(0.5, 0.5)]
        landmarks.extend(FakeLandmark(-1.0, -1.0) for _ in range(20))
        fake_hand = FakeHandLandmarks(landmarks)

        joints = BagToHands._landmarks_to_3d(
            self.processor,
            fake_hand,
            depth_image=depth,
            W=10,
            H=10,
            fx=2.0,
            fy=2.0,
            cx=0.0,
            cy=0.0,
        )

        self.assertEqual(joints.shape, (21, 3))
        np.testing.assert_allclose(joints[0], np.array([2.5, 2.5, 1.0], dtype=np.float32), atol=1e-6)
        self.assertTrue(np.all(np.isnan(joints[1:])))

    def test_compute_spike_mask_flags_large_velocity_jump(self) -> None:
        times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        positions = np.full((3, 21, 3), np.nan, dtype=np.float32)
        positions[0, 0] = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        positions[1, 0] = np.array([2.0, 0.0, 0.5], dtype=np.float32)
        positions[2, 0] = np.array([2.1, 0.0, 0.5], dtype=np.float32)

        mask = BagToHands._compute_spike_mask(positions, times, max_speed_mps=4.0)

        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[2, 0])
        self.assertEqual(int(mask.sum()), 1)


if __name__ == "__main__":
    unittest.main()
