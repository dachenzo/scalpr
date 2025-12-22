import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
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

        self.point_color = "#3498db"
        self.line_color = "#1f5fa3"

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
            times = data["times"]

        T = positions.shape[0]
        if T == 0:
            raise ValueError(f"No frames in input npz: {self.npz_path}")

        frame_indices = np.arange(0, T, self.frame_stride)
        if frame_indices.size == 0:
            raise ValueError(f"No frames selected by frame_stride={self.frame_stride}")

        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        fps = float(np.clip(30.0 / self.frame_stride, 1.0, 60.0))
        interval_ms = int(round(1000.0 / fps))

        self._render_skeleton_video(
            positions=positions,
            times=times,
            frame_indices=frame_indices,
            hand_edges=hand_edges,
            interval_ms=interval_ms,
            fps=fps,
        )

    def _render_skeleton_video(
        self,
        positions: np.ndarray,
        times: np.ndarray,
        frame_indices: np.ndarray,
        hand_edges: list[tuple[int, int]],
        interval_ms: int,
        fps: float,
    ) -> None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")
        self._configure_3d_axes(ax, positions)

        scatter = ax.scatter([], [], [], s=80, c=self.point_color)
        lines = [ax.plot([], [], [], linewidth=3.0, c=self.line_color)[0] for _ in hand_edges]

        def update(frame_idx: int):
            P = positions[frame_idx]
            finite = np.all(np.isfinite(P), axis=1)

            scatter._offsets3d = (P[finite, 0], P[finite, 1], P[finite, 2])

            for i, (a, b) in enumerate(hand_edges):
                if finite[a] and finite[b]:
                    lines[i].set_data([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]])
                    lines[i].set_3d_properties([P[a, 2], P[b, 2]])
                else:
                    lines[i].set_data([], [])
                    lines[i].set_3d_properties([])

            ax.set_title(f"Frame {frame_idx}, t={times[frame_idx]:.2f}s")
            return [scatter, *lines]

        anim = FuncAnimation(fig, update, frames=frame_indices, interval=interval_ms, blit=False)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(self.output_path, writer=FFMpegWriter(fps=fps))
        plt.close(fig)
        print(f"Saved animation to {self.output_path}")

    @staticmethod
    def _configure_3d_axes(ax, positions: np.ndarray) -> None:
        finite = np.isfinite(positions)
        if not finite.any():
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.set_zlim(0.0, 0.6)
        else:
            all_x = positions[:, :, 0]
            all_y = positions[:, :, 1]
            all_z = positions[:, :, 2]
            x_min, x_max = np.nanpercentile(all_x, [5, 95])
            y_min, y_max = np.nanpercentile(all_y, [5, 95])
            z_min, z_max = np.nanpercentile(all_z, [5, 95])

            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            z_mid = (z_min + z_max) / 2
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            if max_range <= 0:
                max_range = 1e-3
            half = (max_range / 2) * 1.05

            ax.set_xlim(x_mid - half, x_mid + half)
            ax.set_ylim(y_mid - half, y_mid + half)
            ax.set_zlim(z_mid - half, z_mid + half)

        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")


if __name__ == "__main__":
    pass
