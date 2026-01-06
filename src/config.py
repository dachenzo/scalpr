from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppConfig:
    bag_path: str
    output_dir: str = "out"
    frame_stride: int = 1
    viz_frame_stride: int = 2
    timeout_ms: int = 3000


def _read_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def _value(cli_value: Any, cfg_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default


def load_config(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Extract 3D hand joints from a RealSense .bag and render motion videos."
    )
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--bag-path", help="Input RealSense .bag file path.")
    parser.add_argument("--output-dir", help="Directory for generated npz/mp4 outputs.")
    parser.add_argument("--frame-stride", type=int, help="Frame stride for bag processing.")
    parser.add_argument("--viz-frame-stride", type=int, help="Frame stride for visualization animation.")
    parser.add_argument("--timeout-ms", type=int, help="Frame read timeout in milliseconds.")

    args = parser.parse_args(argv)
    cfg = _read_json_config(args.config)

    bag_path = _value(args.bag_path, cfg.get("bag_path"), None)
    if not bag_path:
        parser.error("Bag path is required. Set --bag-path or config['bag_path'].")

    output_dir = _value(args.output_dir, cfg.get("output_dir"), "out")
    frame_stride = int(_value(args.frame_stride, cfg.get("frame_stride"), 1))
    viz_frame_stride = int(_value(args.viz_frame_stride, cfg.get("viz_frame_stride"), 2))
    timeout_ms = int(_value(args.timeout_ms, cfg.get("timeout_ms"), 3000))

    return AppConfig(
        bag_path=bag_path,
        output_dir=output_dir,
        frame_stride=max(1, frame_stride),
        viz_frame_stride=max(1, viz_frame_stride),
        timeout_ms=max(1, timeout_ms),
    )
