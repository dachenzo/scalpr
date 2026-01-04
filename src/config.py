from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    bag_path: str
    output_dir: str = "out"
    frame_stride: int = 1
    viz_frame_stride: int = 2
    timeout_ms: int = 3000


def load_config(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Extract 3D hand joints from a RealSense .bag and render motion videos."
    )
    parser.add_argument("--bag-path", help="Input RealSense .bag file path.")
    parser.add_argument("--output-dir", help="Directory for generated npz/mp4 outputs.")
    parser.add_argument("--frame-stride", type=int, help="Frame stride for bag processing.")
    parser.add_argument("--viz-frame-stride", type=int, help="Frame stride for visualization animation.")
    parser.add_argument("--timeout-ms", type=int, help="Frame read timeout in milliseconds.")

    args = parser.parse_args(argv)

    if not args.bag_path:
        parser.error("Bag path is required. Set --bag-path.")

    return AppConfig(
        bag_path=args.bag_path,
        output_dir=args.output_dir or "out",
        frame_stride=max(1, int(args.frame_stride or 1)),
        viz_frame_stride=max(1, int(args.viz_frame_stride or 2)),
        timeout_ms=max(1, int(args.timeout_ms or 3000)),
    )
