import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
from pathlib import Path


class BagToHands:
    MAX_HANDS = 2
    HAND_DETECTION_CONFIDENCE = 0.7
    HAND_TRACKING_CONFIDENCE = 0.5
    TIMEOUT = 3000  # ms

    """
    Process a RealSense .bag file to extract 3D hand joint positions using MediaPipe Hands.
    """

    def __init__(
        self,
        bag_path: str,
        left_output_path: str = "out/hand_3d_left.npz",
        right_output_path: str = "out/hand_3d_right.npz",
        frame_stride: int = 1,
        timeout_ms: int = TIMEOUT,
    ):
        self.path = bag_path
        self.left_output_path = left_output_path
        self.right_output_path = right_output_path
        self.frame_stride = max(1, int(frame_stride))
        self.timeout_ms = max(1, int(timeout_ms))

        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(bag_path, repeat_playback=False)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_intrinsics = depth_stream.get_intrinsics()
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.MAX_HANDS,
            min_detection_confidence=self.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.HAND_TRACKING_CONFIDENCE,
        )

    # Replace single-pixel depth lookup with a small window median
    def _get_depth_at(self, depth_image: np.ndarray, u: int, v: int, window: int = 2) -> float:
        h, w = depth_image.shape
        u0, u1 = max(0, u - window), min(w, u + window + 1)
        v0, v1 = max(0, v - window), min(h, v + window + 1)
        patch = depth_image[v0:v1, u0:u1]
        valid = patch[patch > 0]
        return float(np.median(valid)) if valid.size > 0 else 0.0
    
    # 2D landmarks + depth -> 3D joints
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

            d_raw = self._get_depth_at(depth_image, u, v)
            if d_raw == 0:
                continue

            d = d_raw * self.depth_scale  # meters
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
            joints_3d[j] = (X, Y, Z)

        return joints_3d

    def run(self):
        frame_times = []
        left_raw_positions = []
        right_raw_positions = []

        fx = self.depth_intrinsics.fx
        fy = self.depth_intrinsics.fy
        cx = self.depth_intrinsics.ppx
        cy = self.depth_intrinsics.ppy

        playback = self.profile.get_device().as_playback()
        playback.set_real_time(False)

        frame_idx = 0

        try:
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
                except RuntimeError:
                    break  # end of bag / timeout

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

                result = self.hands.process(color_image)

                if result.multi_hand_landmarks:
                    handedness_list = getattr(result, "multi_handedness", None)

                    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                        label = None
                        if handedness_list and i < len(handedness_list):
                            label = handedness_list[i].classification[0].label  # "Left" / "Right"

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
                    print(f"Frame {frame_idx}: no hand detected")

                left_raw_positions.append(left_joints)
                right_raw_positions.append(right_joints)

                frame_idx += 1

        finally:
            self.pipeline.stop()
            self.hands.close()

        Path(self.left_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.right_output_path).parent.mkdir(parents=True, exist_ok=True)

        if len(frame_times) == 0:
            print("No frames were processed; not saving output files.")
            return

        times = np.array(frame_times, dtype=np.float64)
        left_raw = np.stack(left_raw_positions, axis=0)
        right_raw = np.stack(right_raw_positions, axis=0)

        np.savez(self.left_output_path, times=times, raw_positions=left_raw)
        print(f"Saved {self.left_output_path} with {left_raw.shape[0]} frames")

        np.savez(self.right_output_path, times=times, raw_positions=right_raw)
        print(f"Saved {self.right_output_path} with {right_raw.shape[0]} frames")


if __name__ == "__main__":
    bag_path = "/path/to/your/file.bag"
    processor = BagToHands(bag_path)
    processor.run()
