import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
from pathlib import Path


class BagToHands:
    MAX_HANDS = 2
    HAND_DETECTION_CONFIDENCE = 0.7  # might want to lower this for some sequences
    HAND_TRACKING_CONFIDENCE = 0.5
    TIMEOUT = 3000  # ms
    DEPTH_MIN_METERS = 0.08
    DEPTH_MAX_METERS = 2.5
    DEPTH_TRIM_LOW_PCT = 20.0
    DEPTH_TRIM_HIGH_PCT = 80.0
    DESPIKE_MAX_SPEED_MPS = 4.0

    """
    Process a RealSense .bag file to extract 3D hand joint positions using MediaPipe Hands.
    Saves the results to .npz files with 'times' and 'positions' arrays for left and right hands.
    """

    def __init__(
        self,
        bag_path: str,
        left_output_path: str = "out/hand_3d_left.npz",
        right_output_path: str = "out/hand_3d_right.npz",
        frame_stride: int = 1,
        timeout_ms: int = TIMEOUT,
        smoothing: bool = False,
        smoothing_alpha: float = 0.35,
        fill_mode: str = "interp",
        interpolation_max_gap: int = 3,
        smoothing_method: str = "ema",
        savgol_window: int = 7,
        savgol_polyorder: int = 2,
        despike_max_speed_mps: float = DESPIKE_MAX_SPEED_MPS,
        depth_min_meters: float = DEPTH_MIN_METERS,
        depth_max_meters: float = DEPTH_MAX_METERS,
    ):
        self.path = bag_path
        self.left_output_path = left_output_path
        self.right_output_path = right_output_path
        self.frame_stride = max(1, int(frame_stride))
        self.timeout_ms = max(1, int(timeout_ms))

        self.smoothing = bool(smoothing)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))

        self.fill_mode = str(fill_mode).lower()
        if self.fill_mode not in {"interp", "none"}:
            raise ValueError("fill_mode must be 'interp' or 'none'.")
        self.interpolation_max_gap = max(0, int(interpolation_max_gap))

        self.smoothing_method = str(smoothing_method).lower()
        if self.smoothing_method not in {"ema", "savgol"}:
            raise ValueError("smoothing_method must be 'ema' or 'savgol'.")
        self.savgol_window = max(3, int(savgol_window))
        self.savgol_polyorder = max(1, int(savgol_polyorder))
        self.despike_max_speed_mps = max(0.0, float(despike_max_speed_mps))
        self.depth_min_meters = max(0.0, float(depth_min_meters))
        self.depth_max_meters = max(self.depth_min_meters, float(depth_max_meters))

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(bag_path, repeat_playback=False)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_intrinsics = depth_stream.get_intrinsics()
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # e.g. 0.001 for mm->m

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.MAX_HANDS,
            min_detection_confidence=self.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.HAND_TRACKING_CONFIDENCE,
        )

    def _get_depth_at(self, depth_image: np.ndarray, u: int, v: int, window: int = 2) -> float:
        h, w = depth_image.shape
        if not (0 <= u < w and 0 <= v < h):
            return float("nan")

        u0, u1 = max(0, u - window), min(w, u + window + 1)
        v0, v1 = max(0, v - window), min(h, v + window + 1)
        patch = depth_image[v0:v1, u0:u1].astype(np.float32) * self.depth_scale

        valid = np.isfinite(patch)
        valid &= patch > self.depth_min_meters
        valid &= patch < self.depth_max_meters
        values = patch[valid]
        if values.size < 3:
            return float("nan")

        lo, hi = np.percentile(values, [self.DEPTH_TRIM_LOW_PCT, self.DEPTH_TRIM_HIGH_PCT])
        trimmed = values[(values >= lo) & (values <= hi)]
        if trimmed.size == 0:
            return float("nan")

        return float(np.median(trimmed))

    def _landmarks_to_3d(
        self,
        hand_landmarks,
        depth_image: np.ndarray,
        W: int,
        H: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        joints_3d = np.full((21, 3), np.nan, dtype=np.float32)

        for j, lm in enumerate(hand_landmarks.landmark):
            u = int(lm.x * W)
            v = int(lm.y * H)

            if not (0 <= u < W and 0 <= v < H):
                continue

            d_m = self._get_depth_at(depth_image, u, v)
            if not np.isfinite(d_m) or d_m <= 0:
                continue

            d = d_m
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
            joints_3d[j] = (X, Y, Z)

        return joints_3d

    def run(self):
        frame_times = []
        left_raw_positions = []
        right_raw_positions = []

        playback = self.profile.get_device().as_playback()
        playback.set_real_time(False)

        frame_idx = 0

        try:
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
                except RuntimeError:
                    break

                if frame_idx % self.frame_stride != 0:
                    frame_idx += 1
                    continue

                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    frame_idx += 1
                    continue

                t_sec = depth_frame.get_timestamp() / 1000.0
                frame_times.append(t_sec)

                left_joints = np.full((21, 3), np.nan, dtype=np.float32)
                right_joints = np.full((21, 3), np.nan, dtype=np.float32)

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                H, W, _ = color_image.shape

                aligned_depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                fx = aligned_depth_intrinsics.fx
                fy = aligned_depth_intrinsics.fy
                cx = aligned_depth_intrinsics.ppx
                cy = aligned_depth_intrinsics.ppy

                result = self.hands.process(color_image)

                if result.multi_hand_landmarks:
                    handedness_list = getattr(result, "multi_handedness", None)

                    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                        label = None
                        if handedness_list and i < len(handedness_list):
                            label = handedness_list[i].classification[0].label

                        joints_3d = self._landmarks_to_3d(
                            hand_landmarks, depth_image, W, H, fx, fy, cx, cy
                        )

                        if not np.isfinite(joints_3d).any():
                            continue

                        if label == "Left":
                            left_joints = joints_3d
                        elif label == "Right":
                            right_joints = joints_3d
                        else:
                            print(f"Frame {frame_idx}: hand with unknown label, skipping")
                else:
                    print(f"Frame {frame_idx}: no hand detected")

                left_raw_positions.append(left_joints)
                right_raw_positions.append(right_joints)

                frame_idx += 1

        finally:
            self.pipeline.stop()
            self.hands.close()

        Path(self.left_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.right_output_path).parent.mkdir(parents=True, exist_ok=True)

        frame_count = len(frame_times)
        if frame_count == 0:
            print("No frames were processed; not saving output files.")
            return

        times = np.array(frame_times, dtype=np.float64)
        left_raw = np.stack(left_raw_positions, axis=0)
        right_raw = np.stack(right_raw_positions, axis=0)

        left_spike_mask = self._compute_spike_mask(left_raw, times, self.despike_max_speed_mps)
        right_spike_mask = self._compute_spike_mask(right_raw, times, self.despike_max_speed_mps)
        left_despiked = self._apply_spike_mask(left_raw, left_spike_mask)
        right_despiked = self._apply_spike_mask(right_raw, right_spike_mask)

        left_spike_events = int(np.sum(left_spike_mask))
        right_spike_events = int(np.sum(right_spike_mask))
        if left_spike_events > 0 or right_spike_events > 0:
            print(
                "Despike events: "
                f"left={left_spike_events}, right={right_spike_events}, "
                f"max_speed={self.despike_max_speed_mps:.2f} m/s"
            )

        if self.fill_mode == "interp":
            left_filled = self._interpolate_short_gaps(left_despiked, self.interpolation_max_gap)
            right_filled = self._interpolate_short_gaps(right_despiked, self.interpolation_max_gap)
        else:
            left_filled = left_despiked.copy()
            right_filled = right_despiked.copy()

        if self.smoothing:
            left_smoothed = self._apply_smoothing(left_filled)
            right_smoothed = self._apply_smoothing(right_filled)
        else:
            left_smoothed = left_filled.copy()
            right_smoothed = right_filled.copy()

        left_post_spike_mask = self._compute_spike_mask(left_smoothed, times, self.despike_max_speed_mps)
        right_post_spike_mask = self._compute_spike_mask(right_smoothed, times, self.despike_max_speed_mps)
        left_final = self._apply_spike_mask(left_smoothed, left_post_spike_mask)
        right_final = self._apply_spike_mask(right_smoothed, right_post_spike_mask)

        if self.fill_mode == "interp":
            left_final = self._interpolate_short_gaps(left_final, self.interpolation_max_gap)
            right_final = self._interpolate_short_gaps(right_final, self.interpolation_max_gap)

        left_final_guard_mask = self._compute_spike_mask(left_final, times, self.despike_max_speed_mps)
        right_final_guard_mask = self._compute_spike_mask(right_final, times, self.despike_max_speed_mps)
        left_final = self._apply_spike_mask(left_final, left_final_guard_mask)
        right_final = self._apply_spike_mask(right_final, right_final_guard_mask)

        left_post_spike_events = int(np.sum(left_post_spike_mask))
        right_post_spike_events = int(np.sum(right_post_spike_mask))
        if left_post_spike_events > 0 or right_post_spike_events > 0:
            print(
                "Post-smoothing despike events: "
                f"left={left_post_spike_events}, right={right_post_spike_events}, "
                f"max_speed={self.despike_max_speed_mps:.2f} m/s"
            )

        left_guard_events = int(np.sum(left_final_guard_mask))
        right_guard_events = int(np.sum(right_final_guard_mask))
        if left_guard_events > 0 or right_guard_events > 0:
            print(
                "Final guard despike events: "
                f"left={left_guard_events}, right={right_guard_events}, "
                f"max_speed={self.despike_max_speed_mps:.2f} m/s"
            )

        np.savez(
            self.left_output_path,
            times=times,
            positions=left_final,
            raw_positions=left_raw,
            despiked_positions=left_despiked,
            spike_mask=left_spike_mask,
            filled_positions=left_filled,
            smoothed_positions=left_final,
            pre_post_despike_smoothed_positions=left_smoothed,
            post_smoothing_spike_mask=left_post_spike_mask,
            final_guard_spike_mask=left_final_guard_mask,
            source_frame_stride=self.frame_stride,
            fill_mode=self.fill_mode,
            interpolation_max_gap=self.interpolation_max_gap,
            smoothing_enabled=self.smoothing,
            smoothing_method=self.smoothing_method,
            smoothing_alpha=self.smoothing_alpha,
            savgol_window=self.savgol_window,
            savgol_polyorder=self.savgol_polyorder,
        )
        print(f"Saved {self.left_output_path} with {left_final.shape[0]} frames")

        np.savez(
            self.right_output_path,
            times=times,
            positions=right_final,
            raw_positions=right_raw,
            despiked_positions=right_despiked,
            spike_mask=right_spike_mask,
            filled_positions=right_filled,
            smoothed_positions=right_final,
            pre_post_despike_smoothed_positions=right_smoothed,
            post_smoothing_spike_mask=right_post_spike_mask,
            final_guard_spike_mask=right_final_guard_mask,
            source_frame_stride=self.frame_stride,
            fill_mode=self.fill_mode,
            interpolation_max_gap=self.interpolation_max_gap,
            smoothing_enabled=self.smoothing,
            smoothing_method=self.smoothing_method,
            smoothing_alpha=self.smoothing_alpha,
            savgol_window=self.savgol_window,
            savgol_polyorder=self.savgol_polyorder,
        )
        print(f"Saved {self.right_output_path} with {right_final.shape[0]} frames")

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

    def _apply_smoothing(self, positions: np.ndarray) -> np.ndarray:
        if self.smoothing_method == "ema":
            return self._smooth_positions(positions, self.smoothing_alpha)
        return self._smooth_positions_savgol(positions)

    def _smooth_positions_savgol(self, positions: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import savgol_filter
        except ImportError as exc:
            raise ImportError(
                "savgol smoothing requires scipy. Install scipy or use smoothing_method='ema'."
            ) from exc

        smoothed = positions.copy()
        T = smoothed.shape[0]

        window = min(self.savgol_window, T)
        if window % 2 == 0:
            window = max(3, window - 1)
        polyorder = min(self.savgol_polyorder, window - 1)

        if window < 3 or polyorder < 1:
            return smoothed

        for joint_idx in range(smoothed.shape[1]):
            for dim_idx in range(smoothed.shape[2]):
                series = smoothed[:, joint_idx, dim_idx]
                valid = np.isfinite(series)
                if valid.sum() < window:
                    continue

                interp = series.copy()
                indices = np.arange(T)
                interp[~valid] = np.interp(indices[~valid], indices[valid], series[valid])
                filtered = savgol_filter(interp, window_length=window, polyorder=polyorder, mode="interp")
                series[valid] = filtered[valid]
                smoothed[:, joint_idx, dim_idx] = series

        return smoothed

    @staticmethod
    def _interpolate_short_gaps(positions: np.ndarray, max_gap: int) -> np.ndarray:
        if positions.shape[0] <= 2 or max_gap <= 0:
            return positions.copy()

        interpolated = positions.copy()
        T = interpolated.shape[0]

        for joint_idx in range(interpolated.shape[1]):
            joint_series = interpolated[:, joint_idx, :]
            valid = np.all(np.isfinite(joint_series), axis=1)
            valid_indices = np.where(valid)[0]

            if valid_indices.size < 2:
                continue

            for start_idx, end_idx in zip(valid_indices[:-1], valid_indices[1:]):
                gap = end_idx - start_idx - 1
                if gap <= 0 or gap > max_gap:
                    continue

                start_val = joint_series[start_idx]
                end_val = joint_series[end_idx]
                for offset in range(1, gap + 1):
                    ratio = offset / (gap + 1)
                    joint_series[start_idx + offset] = (1.0 - ratio) * start_val + ratio * end_val

            interpolated[:, joint_idx, :] = joint_series

        return interpolated

    @staticmethod
    def _compute_spike_mask(positions: np.ndarray, times: np.ndarray, max_speed_mps: float) -> np.ndarray:
        spike_mask = np.zeros((positions.shape[0], positions.shape[1]), dtype=bool)
        if positions.shape[0] < 2 or max_speed_mps <= 0:
            return spike_mask

        dt = np.diff(times)
        valid_dt = np.isfinite(dt) & (dt > 1e-6)
        if not np.any(valid_dt):
            return spike_mask

        delta = np.diff(positions, axis=0)
        speed = np.linalg.norm(delta, axis=2)
        speed[valid_dt, :] /= dt[valid_dt, np.newaxis]
        speed[~valid_dt, :] = np.nan

        finite_speed = np.isfinite(speed)
        spike_edges = finite_speed & (speed > max_speed_mps)
        spike_mask[1:, :] = spike_edges
        return spike_mask

    @staticmethod
    def _apply_spike_mask(positions: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
        cleaned = positions.copy()
        cleaned[spike_mask] = np.nan
        return cleaned


if __name__ == "__main__":
    bag_path = "/home/michael/school/syseng/20251112_130013.bag"
    processor = BagToHands(bag_path)
    processor.run()
