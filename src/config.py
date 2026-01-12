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
    left_npz_name: str = "hand_3d_left.npz"
    right_npz_name: str = "hand_3d_right.npz"
    left_mp4_name: str = "hand_3d_motion_left.mp4"
    right_mp4_name: str = "hand_3d_motion_right.mp4"
    frame_stride: int = 1
    viz_frame_stride: int = 2
    timeout_ms: int = 3000
    smoothing: bool = False
    smoothing_alpha: float = 0.35
    fill_mode: str = "interp"
    interpolation_max_gap: int = 3
    smoothing_method: str = "ema"
    savgol_window: int = 7
    savgol_polyorder: int = 2
    debug_viz: bool = False
    left_debug_mp4_name: str = "hand_3d_motion_left_debug.mp4"
    right_debug_mp4_name: str = "hand_3d_motion_right_debug.mp4"
    debug_velocity_trails: bool = False
    debug_trail_length: int = 20


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
    parser.add_argument("--left-npz-name", help="Left-hand npz filename.")
    parser.add_argument("--right-npz-name", help="Right-hand npz filename.")
    parser.add_argument("--left-mp4-name", help="Left-hand MP4 filename.")
    parser.add_argument("--right-mp4-name", help="Right-hand MP4 filename.")
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
    parser.add_argument("--smoothing-method", choices=["ema", "savgol"], help="Temporal smoothing method.")
    parser.add_argument("--savgol-window", type=int, help="Savitzky-Golay odd window length.")
    parser.add_argument("--savgol-polyorder", type=int, help="Savitzky-Golay polynomial order.")

    parser.add_argument("--left-debug-mp4-name", help="Left-hand debug MP4 filename.")
    parser.add_argument("--right-debug-mp4-name", help="Right-hand debug MP4 filename.")
    parser.add_argument("--debug-trail-length", type=int, help="Velocity trail length in frames.")

    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument("--debug-viz", dest="debug_viz", action="store_true")
    debug_group.add_argument("--no-debug-viz", dest="debug_viz", action="store_false")
    parser.set_defaults(debug_viz=None)

    trail_group = parser.add_mutually_exclusive_group()
    trail_group.add_argument("--debug-velocity-trails", dest="debug_velocity_trails", action="store_true")
    trail_group.add_argument("--no-debug-velocity-trails", dest="debug_velocity_trails", action="store_false")
    parser.set_defaults(debug_velocity_trails=None)

    args = parser.parse_args(argv)
    cfg = _read_json_config(args.config)

    env = {
        "bag_path": os.getenv("SCALPR_BAG_PATH"),
        "output_dir": os.getenv("SCALPR_OUTPUT_DIR"),
        "left_npz_name": os.getenv("SCALPR_LEFT_NPZ_NAME"),
        "right_npz_name": os.getenv("SCALPR_RIGHT_NPZ_NAME"),
        "left_mp4_name": os.getenv("SCALPR_LEFT_MP4_NAME"),
        "right_mp4_name": os.getenv("SCALPR_RIGHT_MP4_NAME"),
        "frame_stride": os.getenv("SCALPR_FRAME_STRIDE"),
        "viz_frame_stride": os.getenv("SCALPR_VIZ_FRAME_STRIDE"),
        "timeout_ms": os.getenv("SCALPR_TIMEOUT_MS"),
        "fill_mode": os.getenv("SCALPR_FILL_MODE"),
        "interpolation_max_gap": os.getenv("SCALPR_INTERPOLATION_MAX_GAP"),
        "smoothing": os.getenv("SCALPR_SMOOTHING"),
        "smoothing_alpha": os.getenv("SCALPR_SMOOTHING_ALPHA"),
        "smoothing_method": os.getenv("SCALPR_SMOOTHING_METHOD"),
        "savgol_window": os.getenv("SCALPR_SAVGOL_WINDOW"),
        "savgol_polyorder": os.getenv("SCALPR_SAVGOL_POLYORDER"),
        "debug_viz": os.getenv("SCALPR_DEBUG_VIZ"),
        "left_debug_mp4_name": os.getenv("SCALPR_LEFT_DEBUG_MP4_NAME"),
        "right_debug_mp4_name": os.getenv("SCALPR_RIGHT_DEBUG_MP4_NAME"),
        "debug_velocity_trails": os.getenv("SCALPR_DEBUG_VELOCITY_TRAILS"),
        "debug_trail_length": os.getenv("SCALPR_DEBUG_TRAIL_LENGTH"),
    }

    bag_path = _value(args.bag_path, env["bag_path"], cfg.get("bag_path"), None)
    if not bag_path:
        parser.error("Bag path is required. Set --bag-path or SCALPR_BAG_PATH or config['bag_path'].")

    output_dir = _value(args.output_dir, env["output_dir"], cfg.get("output_dir"), "out")
    left_npz_name = _value(args.left_npz_name, env["left_npz_name"], cfg.get("left_npz_name"), "hand_3d_left.npz")
    right_npz_name = _value(
        args.right_npz_name, env["right_npz_name"], cfg.get("right_npz_name"), "hand_3d_right.npz"
    )
    left_mp4_name = _value(
        args.left_mp4_name, env["left_mp4_name"], cfg.get("left_mp4_name"), "hand_3d_motion_left.mp4"
    )
    right_mp4_name = _value(
        args.right_mp4_name, env["right_mp4_name"], cfg.get("right_mp4_name"), "hand_3d_motion_right.mp4"
    )

    frame_stride = int(_value(args.frame_stride, env["frame_stride"], cfg.get("frame_stride"), 1))
    viz_frame_stride = int(_value(args.viz_frame_stride, env["viz_frame_stride"], cfg.get("viz_frame_stride"), 2))
    timeout_ms = int(_value(args.timeout_ms, env["timeout_ms"], cfg.get("timeout_ms"), 3000))
    fill_mode = str(_value(args.fill_mode, env["fill_mode"], cfg.get("fill_mode"), "interp")).lower()
    interpolation_max_gap = int(
        _value(args.interpolation_max_gap, env["interpolation_max_gap"], cfg.get("interpolation_max_gap"), 3)
    )

    smoothing = _parse_bool(_value(args.smoothing, env["smoothing"], cfg.get("smoothing"), False), False)
    smoothing_alpha = float(_value(args.smoothing_alpha, env["smoothing_alpha"], cfg.get("smoothing_alpha"), 0.35))
    smoothing_method = str(_value(args.smoothing_method, env["smoothing_method"], cfg.get("smoothing_method"), "ema")).lower()
    savgol_window = int(_value(args.savgol_window, env["savgol_window"], cfg.get("savgol_window"), 7))
    savgol_polyorder = int(_value(args.savgol_polyorder, env["savgol_polyorder"], cfg.get("savgol_polyorder"), 2))

    debug_viz = _parse_bool(_value(args.debug_viz, env["debug_viz"], cfg.get("debug_viz"), False), False)
    left_debug_mp4_name = _value(
        args.left_debug_mp4_name,
        env["left_debug_mp4_name"],
        cfg.get("left_debug_mp4_name"),
        "hand_3d_motion_left_debug.mp4",
    )
    right_debug_mp4_name = _value(
        args.right_debug_mp4_name,
        env["right_debug_mp4_name"],
        cfg.get("right_debug_mp4_name"),
        "hand_3d_motion_right_debug.mp4",
    )
    debug_velocity_trails = _parse_bool(
        _value(args.debug_velocity_trails, env["debug_velocity_trails"], cfg.get("debug_velocity_trails"), False),
        False,
    )
    debug_trail_length = int(_value(args.debug_trail_length, env["debug_trail_length"], cfg.get("debug_trail_length"), 20))

    # sanitize
    frame_stride = max(1, frame_stride)
    viz_frame_stride = max(1, viz_frame_stride)
    timeout_ms = max(1, timeout_ms)
    if fill_mode not in {"interp", "none"}:
        fill_mode = "interp"
    interpolation_max_gap = max(0, interpolation_max_gap)
    smoothing_alpha = min(max(smoothing_alpha, 0.0), 1.0)
    if smoothing_method not in {"ema", "savgol"}:
        smoothing_method = "ema"
    savgol_window = max(3, savgol_window)
    savgol_polyorder = max(1, savgol_polyorder)
    debug_trail_length = max(2, debug_trail_length)

    return AppConfig(
        bag_path=bag_path,
        output_dir=output_dir,
        left_npz_name=left_npz_name,
        right_npz_name=right_npz_name,
        left_mp4_name=left_mp4_name,
        right_mp4_name=right_mp4_name,
        frame_stride=frame_stride,
        viz_frame_stride=viz_frame_stride,
        timeout_ms=timeout_ms,
        smoothing=smoothing,
        smoothing_alpha=smoothing_alpha,
        fill_mode=fill_mode,
        interpolation_max_gap=interpolation_max_gap,
        smoothing_method=smoothing_method,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        debug_viz=debug_viz,
        left_debug_mp4_name=left_debug_mp4_name,
        right_debug_mp4_name=right_debug_mp4_name,
        debug_velocity_trails=debug_velocity_trails,
        debug_trail_length=debug_trail_length,
    )
