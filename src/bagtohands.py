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
        raise NotImplementedError("run() not implemented yet")


if __name__ == "__main__":
    bag_path = "/path/to/your/file.bag"
    processor = BagToHands(bag_path)
    processor.run()
