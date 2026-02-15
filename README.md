 # scalpr

Process Intel RealSense `.bag` recordings and generate:
- 3D hand joint trajectories (`.npz`) for left and right hands
- motion visualization videos (`.mp4`) for left and right hands

The current pipeline is:
1. Read a RealSense bag file
2. Extract 3D hand landmarks with MediaPipe + depth back-projection
3. Build raw, filled, and smoothed hand tracks
4. Generate a per-run quality report (`run_quality_report.json`)
5. Render left/right motion videos (plus optional debug videos)

## Requirements

- Python `>=3.12,<3.15`
- Intel RealSense bag input (`.bag`)
- `ffmpeg` installed and available on `PATH` (required for MP4 export)

Dependencies are managed in [pyproject.toml](pyproject.toml).

## Quick Start

Install dependencies (example with Poetry):

```bash
poetry install
```

Run the full pipeline:

```bash
poetry run python src/main.py --bag-path /absolute/path/to/input.bag
```

This generates outputs in `out/` by default.

## Configuration Priority

Configuration is loaded in this order (highest to lowest priority):
1. CLI flags
2. Environment variables
3. JSON config file passed with `--config`
4. Built-in defaults

## CLI Usage

```bash
poetry run python src/main.py \
	--bag-path /absolute/path/to/input.bag \
	--output-dir out \
	--frame-stride 1 \
	--viz-frame-stride 2 \
	--timeout-ms 3000 \
	--fill-mode interp \
	--interpolation-max-gap 3 \
	--smoothing \
	--smoothing-alpha 0.35 \
	--debug-viz \
	--debug-velocity-trails \
	--debug-trail-length 20
```

### Main flags

- `--bag-path` (required): input `.bag` file
- `--output-dir`: output directory (default: `out`)
- `--left-npz-name`, `--right-npz-name`: output NPZ filenames
- `--left-mp4-name`, `--right-mp4-name`: output MP4 filenames
- `--frame-stride`: frame sampling for bag processing
- `--viz-frame-stride`: frame sampling for video rendering
- `--timeout-ms`: frame read timeout in milliseconds
- `--smoothing` / `--no-smoothing`: toggle temporal smoothing
- `--smoothing-alpha`: smoothing coefficient in `[0, 1]`
- `--fill-mode`: missing-frame handling (`interp` or `none`)
- `--interpolation-max-gap`: max short gap size for per-joint interpolation
- `--smoothing-method`: smoothing method (`ema` or `savgol`)
- `--savgol-window`, `--savgol-polyorder`: Savitzky–Golay parameters
- `--debug-viz` / `--no-debug-viz`: enable/disable side-by-side debug video
- `--left-debug-mp4-name`, `--right-debug-mp4-name`: debug video filenames
- `--debug-velocity-trails` / `--no-debug-velocity-trails`: wrist trail overlay
- `--debug-trail-length`: trail length in frames

Note: velocity trails are debug-only; regular output videos do not include trails.

## Debug Visualization

By default, the pipeline writes the standard 3D MP4 outputs only.

When `--debug-viz` is enabled, additional debug videos are written that show:
- side-by-side RGB frame and 3D skeleton
- consistent left/right color scheme
- interpolated joints marked with `x`
- optional wrist velocity trail (`--debug-velocity-trails`)

This is implemented as fused MP4 output (not per-frame image dumps) to keep storage manageable.

## Environment Variables

You can run without many CLI flags by setting env vars:

- `SCALPR_BAG_PATH`
- `SCALPR_OUTPUT_DIR`
- `SCALPR_LEFT_NPZ_NAME`
- `SCALPR_RIGHT_NPZ_NAME`
- `SCALPR_LEFT_MP4_NAME`
- `SCALPR_RIGHT_MP4_NAME`
- `SCALPR_FRAME_STRIDE`
- `SCALPR_VIZ_FRAME_STRIDE`
- `SCALPR_TIMEOUT_MS`
- `SCALPR_FILL_MODE`
- `SCALPR_INTERPOLATION_MAX_GAP`
- `SCALPR_SMOOTHING`
- `SCALPR_SMOOTHING_ALPHA`
- `SCALPR_SMOOTHING_METHOD`
- `SCALPR_SAVGOL_WINDOW`
- `SCALPR_SAVGOL_POLYORDER`
- `SCALPR_DEBUG_VIZ`
- `SCALPR_LEFT_DEBUG_MP4_NAME`
- `SCALPR_RIGHT_DEBUG_MP4_NAME`
- `SCALPR_DEBUG_VELOCITY_TRAILS`
- `SCALPR_DEBUG_TRAIL_LENGTH`

Example:

```bash
export SCALPR_BAG_PATH=/absolute/path/to/input.bag
export SCALPR_SMOOTHING=true
poetry run python src/main.py
```

## JSON Config

Example `config.json`:

```json
{
	"bag_path": "/absolute/path/to/input.bag",
	"output_dir": "out",
	"left_npz_name": "hand_3d_left.npz",
	"right_npz_name": "hand_3d_right.npz",
	"left_mp4_name": "hand_3d_motion_left.mp4",
	"right_mp4_name": "hand_3d_motion_right.mp4",
	"left_debug_mp4_name": "hand_3d_motion_left_debug.mp4",
	"right_debug_mp4_name": "hand_3d_motion_right_debug.mp4",
	"frame_stride": 1,
	"viz_frame_stride": 2,
	"timeout_ms": 3000,
	"fill_mode": "interp",
	"interpolation_max_gap": 3,
	"smoothing": true,
	"smoothing_alpha": 0.35,
	"smoothing_method": "ema",
	"savgol_window": 7,
	"savgol_polyorder": 2,
	"debug_viz": false,
	"debug_velocity_trails": false,
	"debug_trail_length": 20
}
```

Run with:

```bash
poetry run python src/main.py --config config.json
```

## Testing

Run the test suite:

```bash
poetry run python -m unittest discover -s tests -v
```

Included tests:
- unit tests for `BagToHands._get_depth_at`
- unit tests for `BagToHands._landmarks_to_3d`
- smoke test for `HandMotionVisualizer` with a tiny synthetic NPZ

## Outputs

By default, `out/` contains:
- `hand_3d_left.npz`
- `hand_3d_right.npz`
- `hand_3d_motion_left.mp4`
- `hand_3d_motion_right.mp4`
- `run_quality_report.json`

If `--debug-viz` is enabled, it also includes:
- `hand_3d_motion_left_debug.mp4`
- `hand_3d_motion_right_debug.mp4`

Each NPZ includes:
- `times`: timestamps
- `raw_positions`: raw frame-aligned hand joints `(T, 21, 3)`
- `filled_positions`: post-gap-fill hand joints `(T, 21, 3)`
- `smoothed_positions`: post-smoothing hand joints `(T, 21, 3)`
- `positions`: current selected output track `(T, 21, 3)`

## Run Quality Report

`src/main.py` automatically invokes [scripts/scaling.py](scripts/scaling.py) after extraction to write:
- `out/run_quality_report.json`

Reported metrics include:
- total frames read
- hand detection rate (left/right, and any-hand when aligned)
- valid-joint percentage (left/right)
- longest missing streak (left/right)
- left/right frame count mismatch
- left/right detected-frame mismatch

The report also includes:
- `status`: `ok` or `warn`
- `warnings`: threshold violations

Default warning thresholds (overridable via CLI):

```bash
poetry run python scripts/scaling.py \
	--left-npz out/hand_3d_left.npz \
	--right-npz out/hand_3d_right.npz \
	--report-out out/run_quality_report.json \
	--min-detection-rate 0.90 \
	--min-valid-joint-percentage 0.90 \
	--max-missing-streak 15 \
	--max-frame-count-mismatch 0 \
	--max-detected-frame-mismatch 30
```

## Camera Setup (To Be Filled)

Add setup photos and guidance here to standardize capture quality.

### Recommended camera position

![Camera position overview](docs/images/camera_position_overview.png)

### Side angle

![Camera side angle](docs/images/camera_side_angle.png)

### Distance and height reference

![Camera distance and height](docs/images/camera_distance_height.png)

### Notes

- Keep camera rigidly mounted (no handheld motion)
- Keep hand workspace fully inside frame
- Avoid strong reflective glare on instruments/gloves
- Keep lighting consistent across sessions