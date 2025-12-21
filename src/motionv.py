import numpy as np
from pathlib import Path


class HandMotionVisualizer:
    """Visualize 3D hand joint positions over time from a .npz file."""

    def __init__(
        self,
        npz_path: str,
        output_path: str = "hand_3d_motion.mp4",
        frame_stride: int = 2,
    ):
        self.npz_path = npz_path
        self.output_path = output_path
        self.frame_stride = max(1, int(frame_stride))

    def run(self):
        if not Path(self.npz_path).exists():
            raise FileNotFoundError(f"Input npz not found: {self.npz_path}")

        with np.load(self.npz_path, allow_pickle=True) as data:
            if "positions" in data.files:
                positions = data["positions"]
            elif "raw_positions" in data.files:
                positions = data["raw_positions"]
            else:
                raise KeyError("Expected 'positions' or 'raw_positions' in npz.")

            if "times" not in data.files:
                raise KeyError("Expected 'times' in npz.")
            times = data["times"]

        if positions.shape[0] == 0:
            raise ValueError(f"No frames in input npz: {self.npz_path}")

        # Pick frames to render
        frame_indices = np.arange(0, positions.shape[0], self.frame_stride)
        if frame_indices.size == 0:
            raise ValueError(f"No frames selected by frame_stride={self.frame_stride}")

        print(f"Loaded {positions.shape[0]} frames, rendering {frame_indices.size} frames to {self.output_path}")


if __name__ == "__main__":
    # Example usage:
    # HandMotionVisualizer("out/hand_3d_left.npz").run()
    pass
