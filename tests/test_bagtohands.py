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

        self.assertEqual(value, 1100)

    def test_get_depth_at_returns_zero_when_patch_has_no_valid_depth(self) -> None:
        depth = np.zeros((5, 5), dtype=np.uint16)

        value = BagToHands._get_depth_at(self.processor, depth, u=2, v=2, window=1)

        self.assertEqual(value, 0)

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


if __name__ == "__main__":
    unittest.main()
