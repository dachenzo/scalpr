from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motionv import HandMotionVisualizer


class MotionVisualizerAxisMapTests(unittest.TestCase):
    def test_axis_map_reorders_and_flips_axes(self) -> None:
        visualizer = HandMotionVisualizer(npz_path="dummy_right.npz", axis_map="x,z,-y")

        positions = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
        transformed = visualizer._apply_axis_transform(positions)

        expected = np.array([[[1.0, 3.0, -2.0], [4.0, 6.0, -5.0]]], dtype=np.float32)
        np.testing.assert_allclose(transformed, expected)
        self.assertEqual(visualizer.axis_labels, ["Xcam", "Zcam", "-Ycam"])


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for MP4 smoke test")
class MotionVisualizerSmokeTests(unittest.TestCase):
    def test_visualizer_writes_mp4_from_tiny_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            npz_path = tmp_path / "tiny_hand.npz"
            mp4_path = tmp_path / "tiny_hand.mp4"

            times = np.array([0.0, 1 / 30, 2 / 30, 3 / 30], dtype=np.float64)
            base = np.linspace(0.0, 0.1, 21, dtype=np.float32)
            positions = np.stack(
                [
                    np.stack([base + (i * 0.001), base * 0.5, np.full_like(base, 0.4 + i * 0.002)], axis=1)
                    for i in range(4)
                ],
                axis=0,
            )

            np.savez(
                npz_path,
                times=times,
                positions=positions,
                raw_positions=positions,
                smoothed_positions=positions,
                source_frame_stride=1,
            )

            visualizer = HandMotionVisualizer(
                npz_path=str(npz_path),
                output_path=str(mp4_path),
                frame_stride=1,
                smoothing=False,
                debug=False,
            )
            visualizer.run()

            self.assertTrue(mp4_path.exists())
            self.assertGreater(mp4_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
