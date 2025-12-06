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

    def run(self):
        raise NotImplementedError("run() not implemented yet")


if __name__ == "__main__":
    bag_path = "/path/to/your/file.bag"
    processor = BagToHands(bag_path)
    processor.run()
