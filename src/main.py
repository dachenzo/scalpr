from pathlib import Path
import subprocess
import sys

from bagtohands import BagToHands
from config import load_config
from motionv import HandMotionVisualizer

if __name__ == "__main__":
    app_config = load_config()
    output_dir = Path(app_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_npz_path = str(output_dir / app_config.left_npz_name)
    right_npz_path = str(output_dir / app_config.right_npz_name)
    left_mp4_path = str(output_dir / app_config.left_mp4_name)
    right_mp4_path = str(output_dir / app_config.right_mp4_name)
    left_debug_mp4_path = str(output_dir / app_config.left_debug_mp4_name)
    right_debug_mp4_path = str(output_dir / app_config.right_debug_mp4_name)

    processor = BagToHands(
        app_config.bag_path,
        left_output_path=left_npz_path,
        right_output_path=right_npz_path,
        frame_stride=app_config.frame_stride,
        timeout_ms=app_config.timeout_ms,
        smoothing=app_config.smoothing,
        smoothing_alpha=app_config.smoothing_alpha,
        fill_mode=app_config.fill_mode,
        interpolation_max_gap=app_config.interpolation_max_gap,
        smoothing_method=app_config.smoothing_method,
        savgol_window=app_config.savgol_window,
        savgol_polyorder=app_config.savgol_polyorder,
    )
    processor.run()

    repo_root = Path(__file__).resolve().parents[1]
    report_json_path = output_dir / "run_quality_report.json"
    report_script_path = repo_root / "scripts" / "scaling.py"
    subprocess.run(
        [
            sys.executable,
            str(report_script_path),
            "--left-npz",
            left_npz_path,
            "--right-npz",
            right_npz_path,
            "--report-out",
            str(report_json_path),
        ],
        check=False,
    )

    visualizer_left = HandMotionVisualizer(
        left_npz_path,
        left_mp4_path,
        frame_stride=app_config.viz_frame_stride,
        smoothing=app_config.smoothing,
        smoothing_alpha=app_config.smoothing_alpha,
        hand_label="left",
        debug=app_config.debug_viz,
        debug_bag_path=app_config.bag_path,
        debug_output_path=left_debug_mp4_path,
        velocity_trails=app_config.debug_velocity_trails,
        trail_length=app_config.debug_trail_length,
    )
    visualizer_left.run()

    visualizer_right = HandMotionVisualizer(
        right_npz_path,
        right_mp4_path,
        frame_stride=app_config.viz_frame_stride,
        smoothing=app_config.smoothing,
        smoothing_alpha=app_config.smoothing_alpha,
        hand_label="right",
        debug=app_config.debug_viz,
        debug_bag_path=app_config.bag_path,
        debug_output_path=right_debug_mp4_path,
        velocity_trails=app_config.debug_velocity_trails,
        trail_length=app_config.debug_trail_length,
    )
    visualizer_right.run()