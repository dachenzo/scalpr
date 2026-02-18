import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


class HandMotionVisualizer:
    """
    Visualize 3D hand joint positions over time from a .npz file.
    """

    def __init__(
        self,
        npz_path: str,
        output_path: str = "hand_3d_motion.mp4",
        frame_stride: int = 2,
        smoothing: bool = False,
        smoothing_alpha: float = 0.35,
        axis_map: str = "x,y,z",
        hand_label: str | None = None,
        debug: bool = False,
        debug_bag_path: str | None = None,
        debug_output_path: str | None = None,
        velocity_trails: bool = False,
        trail_length: int = 20,
    ):
        self.npz_path = npz_path
        self.output_path = output_path
        self.frame_stride = max(1, int(frame_stride))
        self.smoothing = bool(smoothing)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.axis_transform, self.axis_labels = self._parse_axis_map(axis_map)
        self.hand_label = (hand_label or self._infer_hand_label(npz_path)).lower()
        self.debug = bool(debug)
        self.debug_bag_path = debug_bag_path
        self.debug_output_path = debug_output_path
        self.velocity_trails = bool(velocity_trails)
        self.trail_length = max(2, int(trail_length))

        if self.hand_label == "left":
            self.point_color = "#e74c3c"
            self.line_color = "#c0392b"
            self.trail_color = "#f39c12"
        else:
            self.point_color = "#3498db"
            self.line_color = "#1f5fa3"
            self.trail_color = "#9b59b6"
        self.interp_color = "#f1c40f"

    def run(self):
        if not Path(self.npz_path).exists():
            raise FileNotFoundError(f"Input npz not found: {self.npz_path}")

        with np.load(self.npz_path, allow_pickle=True) as data:
            data_keys = set(data.files)
            if self.smoothing and "smoothed_positions" in data_keys:
                positions = data["smoothed_positions"]
            elif "positions" in data_keys:
                positions = data["positions"]
            else:
                positions = data["raw_positions"]
            times = data["times"]
            raw_positions = data["raw_positions"] if "raw_positions" in data_keys else None
            source_frame_stride = int(data["source_frame_stride"]) if "source_frame_stride" in data_keys else 1

        if self.smoothing and "smoothed_positions" not in data_keys:
            positions = self._smooth_positions(positions, self.smoothing_alpha)

        positions = self._apply_axis_transform(positions)
        if raw_positions is not None:
            raw_positions = self._apply_axis_transform(raw_positions)

        T = positions.shape[0]
        if T == 0:
            raise ValueError(f"No frames in input npz: {self.npz_path}")

        frame_indices = np.arange(0, T, self.frame_stride)
        if frame_indices.size == 0:
            raise ValueError(f"No frames selected by frame_stride={self.frame_stride} for {self.npz_path}")

        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        dt = np.diff(times[frame_indices]) if frame_indices.size > 1 else np.array([], dtype=np.float64)
        valid_dt = dt[np.isfinite(dt) & (dt > 0)]
        if valid_dt.size > 0:
            fps = float(np.clip(1.0 / np.median(valid_dt), 1.0, 60.0))
        else:
            fps = float(np.clip(30.0 / self.frame_stride, 1.0, 60.0))
        interval_ms = int(round(1000.0 / fps))

        self._render_skeleton_video(
            positions=positions,
            raw_positions=raw_positions,
            times=times,
            frame_indices=frame_indices,
            hand_edges=hand_edges,
            interval_ms=interval_ms,
            fps=fps,
        )

        if self.debug:
            if not self.debug_bag_path:
                raise ValueError("debug_bag_path is required when debug=True.")

            debug_output = self.debug_output_path
            if not debug_output:
                output = Path(self.output_path)
                debug_output = str(output.with_name(f"{output.stem}_debug{output.suffix}"))

            rgb_frames = self._load_debug_color_frames(
                bag_path=self.debug_bag_path,
                total_processed_frames=T,
                source_frame_stride=source_frame_stride,
                frame_indices=frame_indices,
            )

            self._render_debug_video(
                positions=positions,
                raw_positions=raw_positions,
                times=times,
                frame_indices=frame_indices,
                rgb_frames=rgb_frames,
                hand_edges=hand_edges,
                interval_ms=interval_ms,
                fps=fps,
                output_path=debug_output,
            )

    def _render_skeleton_video(
        self,
        positions: np.ndarray,
        raw_positions: np.ndarray | None,
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
        interp_scatter = ax.scatter([], [], [], s=95, c=self.interp_color, marker="x")
        lines = [ax.plot([], [], [], linewidth=3.5, c=self.line_color)[0] for _ in hand_edges]

        def update(frame_idx: int):
            P = positions[frame_idx]
            finite = np.all(np.isfinite(P), axis=1)
            scatter._offsets3d = (P[finite, 0], P[finite, 1], P[finite, 2])

            interp_count = 0
            if raw_positions is not None and frame_idx < raw_positions.shape[0]:
                raw = raw_positions[frame_idx]
                interp_mask = (~np.all(np.isfinite(raw), axis=1)) & finite
                interp_scatter._offsets3d = (
                    P[interp_mask, 0],
                    P[interp_mask, 1],
                    P[interp_mask, 2],
                )
                interp_count = int(interp_mask.sum())
            else:
                interp_scatter._offsets3d = ([], [], [])

            for i, (a, b) in enumerate(hand_edges):
                if finite[a] and finite[b]:
                    lines[i].set_data([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]])
                    lines[i].set_3d_properties([P[a, 2], P[b, 2]])
                else:
                    lines[i].set_data([], [])
                    lines[i].set_3d_properties([])

            ax.set_title(
                f"{self.hand_label.title()} Frame {frame_idx}, t={times[frame_idx]:.2f}s | interp joints: {interp_count}"
            )
            return [scatter, interp_scatter, *lines]

        anim = FuncAnimation(fig, update, frames=frame_indices, interval=interval_ms, blit=False)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(self.output_path, writer=FFMpegWriter(fps=fps))
        plt.close(fig)
        print(f"Saved animation to {self.output_path}")

    def _render_debug_video(
        self,
        positions: np.ndarray,
        raw_positions: np.ndarray | None,
        times: np.ndarray,
        frame_indices: np.ndarray,
        rgb_frames: list[np.ndarray],
        hand_edges: list[tuple[int, int]],
        interval_ms: int,
        fps: float,
        output_path: str,
    ) -> None:
        fig = plt.figure(figsize=(16, 8))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
        self._configure_3d_axes(ax_3d, positions)

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        first = rgb_frames[0] if rgb_frames and rgb_frames[0] is not None else blank
        image_artist = ax_img.imshow(first)
        ax_img.axis("off")
        ax_img.set_title("RGB Frame")

        scatter = ax_3d.scatter([], [], [], s=70, c=self.point_color)
        interp_scatter = ax_3d.scatter([], [], [], s=85, c=self.interp_color, marker="x")
        lines = [ax_3d.plot([], [], [], linewidth=3.0, c=self.line_color)[0] for _ in hand_edges]

        trail_line = (
            ax_3d.plot([], [], [], linewidth=2.0, c=self.trail_color, alpha=0.9)[0]
            if self.velocity_trails
            else None
        )
        trail_points: list[np.ndarray] = []

        def update(k: int):
            frame_idx = int(frame_indices[k])
            rgb = rgb_frames[k] if k < len(rgb_frames) and rgb_frames[k] is not None else blank
            image_artist.set_data(rgb)

            P = positions[frame_idx]
            finite = np.all(np.isfinite(P), axis=1)
            scatter._offsets3d = (P[finite, 0], P[finite, 1], P[finite, 2])

            interp_count = 0
            if raw_positions is not None and frame_idx < raw_positions.shape[0]:
                raw = raw_positions[frame_idx]
                interp_mask = (~np.all(np.isfinite(raw), axis=1)) & finite
                interp_scatter._offsets3d = (
                    P[interp_mask, 0],
                    P[interp_mask, 1],
                    P[interp_mask, 2],
                )
                interp_count = int(interp_mask.sum())
            else:
                interp_scatter._offsets3d = ([], [], [])

            for i, (a, b) in enumerate(hand_edges):
                if finite[a] and finite[b]:
                    lines[i].set_data([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]])
                    lines[i].set_3d_properties([P[a, 2], P[b, 2]])
                else:
                    lines[i].set_data([], [])
                    lines[i].set_3d_properties([])

            artists = [image_artist, scatter, interp_scatter, *lines]

            if trail_line is not None:
                wrist = P[0]
                if np.all(np.isfinite(wrist)):
                    trail_points.append(wrist.copy())
                    if len(trail_points) > self.trail_length:
                        del trail_points[:-self.trail_length]

                if trail_points:
                    trail = np.stack(trail_points, axis=0)
                    trail_line.set_data(trail[:, 0], trail[:, 1])
                    trail_line.set_3d_properties(trail[:, 2])
                else:
                    trail_line.set_data([], [])
                    trail_line.set_3d_properties([])

                artists.append(trail_line)

            ax_3d.set_title(
                f"{self.hand_label.title()} 3D | t={times[frame_idx]:.2f}s | interp joints: {interp_count}"
            )
            return artists

        anim = FuncAnimation(fig, update, frames=range(frame_indices.size), interval=interval_ms, blit=False)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(output_path, writer=FFMpegWriter(fps=fps))
        plt.close(fig)
        print(f"Saved debug animation to {output_path}")

    def _load_debug_color_frames(
        self,
        bag_path: str,
        total_processed_frames: int,
        source_frame_stride: int,
        frame_indices: np.ndarray,
    ) -> list[np.ndarray]:
        import pyrealsense2 as rs

        source_frame_stride = max(1, int(source_frame_stride))
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_path, repeat_playback=False)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        collected: list[np.ndarray | None] = []
        bag_idx = 0
        try:
            while len(collected) < total_processed_frames:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=3000)
                except RuntimeError:
                    break

                if bag_idx % source_frame_stride != 0:
                    bag_idx += 1
                    continue

                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    collected.append(np.asanyarray(color_frame.get_data()).copy())
                bag_idx += 1
        finally:
            pipeline.stop()

        selected: list[np.ndarray] = []
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        for idx in frame_indices:
            if idx < len(collected) and collected[idx] is not None:
                selected.append(collected[idx])
            else:
                selected.append(blank.copy())
        return selected

    @staticmethod
    def _infer_hand_label(npz_path: str) -> str:
        lower = npz_path.lower()
        if "left" in lower:
            return "left"
        if "right" in lower:
            return "right"
        return "right"

    def _configure_3d_axes(self, ax, positions: np.ndarray) -> None:
        x_label, y_label, z_label = self.axis_labels
        finite = np.isfinite(positions)
        if not finite.any():
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.set_zlim(0.0, 0.6)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel(f"{x_label} (m)")
            ax.set_ylabel(f"{y_label} (m)")
            ax.set_zlabel(f"{z_label} (m)")
            return

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
        ax.set_xlabel(f"{x_label} (m)")
        ax.set_ylabel(f"{y_label} (m)")
        ax.set_zlabel(f"{z_label} (m)")

    @staticmethod
    def _parse_axis_component(token: str) -> tuple[int, float, str]:
        token = token.strip().lower()
        sign = 1.0
        if token.startswith("+"):
            token = token[1:].strip()
        elif token.startswith("-"):
            sign = -1.0
            token = token[1:].strip()

        axis_to_index = {"x": 0, "y": 1, "z": 2}
        if token not in axis_to_index:
            raise ValueError("axis_map entries must be one of x, y, z with optional +/- sign.")

        axis_name = token.upper()
        label = f"{'-' if sign < 0 else ''}{axis_name}cam"
        return axis_to_index[token], sign, label

    @classmethod
    def _parse_axis_map(cls, axis_map: str) -> tuple[list[tuple[int, float]], list[str]]:
        parts = [part.strip() for part in str(axis_map).split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError("axis_map must contain exactly three comma-separated entries, e.g. 'x,z,-y'.")

        parsed = [cls._parse_axis_component(part) for part in parts]
        source_indices = [index for index, _, _ in parsed]
        if sorted(source_indices) != [0, 1, 2]:
            raise ValueError("axis_map must use each source axis exactly once.")

        transform = [(index, sign) for index, sign, _ in parsed]
        labels = [label for _, _, label in parsed]
        return transform, labels

    def _apply_axis_transform(self, positions: np.ndarray) -> np.ndarray:
        transformed = np.empty_like(positions)
        for out_axis, (source_axis, sign) in enumerate(self.axis_transform):
            transformed[..., out_axis] = sign * positions[..., source_axis]
        return transformed

    @staticmethod
    def _smooth_positions(positions: np.ndarray, alpha: float) -> np.ndarray:
        if positions.shape[0] <= 1:
            return positions

        smoothed = positions.copy()
        for t in range(1, smoothed.shape[0]):
            prev = smoothed[t - 1]
            curr = smoothed[t]

            valid_curr = np.isfinite(curr)
            valid_prev = np.isfinite(prev)
            blend_mask = valid_curr & valid_prev

            smoothed[t][blend_mask] = alpha * curr[blend_mask] + (1.0 - alpha) * prev[blend_mask]

        return smoothed


if __name__ == "__main__":
    pass
