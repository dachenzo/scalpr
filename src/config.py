from __future__ import annotations

import argparse
import json
import os
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
    smoothing: bool = False
    smoothing_alpha: float = 0.35
    fill_mode: str = "interp"
    interpolation_max_gap: int = 3


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


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


def _value(cli_value: Any, env_value: Any, cfg_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if env_value is not None:
        return env_value
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
    parser.add_argument("--fill-mode", choices=["interp", "none"], help="Missing-frame handling mode.")
    parser.add_argument("--interpolation-max-gap", type=int, help="Maximum gap (frames) to linearly interpolate.")

    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument("--smoothing", dest="smoothing", action="store_true")
    smoothing_group.add_argument("--no-smoothing", dest="smoothing", action="store_false")
    parser.set_defaults(smoothing=None)
    parser.add_argument("--smoothing-alpha", type=float, help="EMA smoothing alpha in [0, 1].")

    args = parser.parse_args(argv)
    cfg = _read_json_config(args.config)

    env = {
        "bag_path": os.getenv("SCALPR_BAG_PATH"),
        "output_dir": os.getenv("SCALPR_OUTPUT_DIR"),
        "frame_stride": os.getenv("SCALPR_FRAME_STRIDE"),
        "viz_frame_stride": os.getenv("SCALPR_VIZ_FRAME_STRIDE"),
        "timeout_ms": os.getenv("SCALPR_TIMEOUT_MS"),
        "fill_mode": os.getenv("SCALPR_FILL_MODE"),
        "interpolation_max_gap": os.getenv("SCALPR_INTERPOLATION_MAX_GAP"),
        "smoothing": os.getenv("SCALPR_SMOOTHING"),
        "smoothing_alpha": os.getenv("SCALPR_SMOOTHING_ALPHA"),
    }

    bag_path = _value(args.bag_path, env["bag_path"], cfg.get("bag_path"), None)
    if not bag_path:
        parser.error("Bag path is required. Set --bag-path or SCALPR_BAG_PATH or config['bag_path'].")

    output_dir = _value(args.output_dir, env["output_dir"], cfg.get("output_dir"), "out")
    frame_stride = int(_value(args.frame_stride, env["frame_stride"], cfg.get("frame_stride"), 1))
    viz_frame_stride = int(_value(args.viz_frame_stride, env["viz_frame_stride"], cfg.get("viz_frame_stride"), 2))
    timeout_ms = int(_value(args.timeout_ms, env["timeout_ms"], cfg.get("timeout_ms"), 3000))
    fill_mode = str(_value(args.fill_mode, env["fill_mode"], cfg.get("fill_mode"), "interp")).lower()
    interpolation_max_gap = int(
        _value(args.interpolation_max_gap, env["interpolation_max_gap"], cfg.get("interpolation_max_gap"), 3)
    )

    smoothing = _parse_bool(_value(args.smoothing, env["smoothing"], cfg.get("smoothing"), False), False)
    smoothing_alpha = float(_value(args.smoothing_alpha, env["smoothing_alpha"], cfg.get("smoothing_alpha"), 0.35))

    # sanitize
    frame_stride = max(1, frame_stride)
    viz_frame_stride = max(1, viz_frame_stride)
    timeout_ms = max(1, timeout_ms)
    if fill_mode not in {"interp", "none"}:
        fill_mode = "interp"
    interpolation_max_gap = max(0, interpolation_max_gap)
    smoothing_alpha = min(max(smoothing_alpha, 0.0), 1.0)

    return AppConfig(
        bag_path=bag_path,
        output_dir=output_dir,
        frame_stride=frame_stride,
        viz_frame_stride=viz_frame_stride,
        timeout_ms=timeout_ms,
        smoothing=smoothing,
        smoothing_alpha=smoothing_alpha,
        fill_mode=fill_mode,
        interpolation_max_gap=interpolation_max_gap,
    )
