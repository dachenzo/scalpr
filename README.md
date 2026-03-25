# Scalpr - 3D Hand Trajectory Extraction Pipeline

Scalpr is an independent subsystem of the COMP0016 Team 18 Hawkes Data Pipeline project.

### What it does

Given a RealSense `.bag` file containing synchronised RGB and depth streams, the pipeline:

1. Detects 2D hand landmarks using MediaPipe Hands
2. Reconstructs 3D joint positions (in meters) via depth back-projection
3. Cleans the trajectories (spike rejection, gap interpolation, optional smoothing)
4. Writes per-hand `.npz` trajectory files and `.mp4` skeleton animations
5. Generates a quality report flagging detection issues

### Sample output

<video src="docs/hand_video.mp4" controls width="600">
  Your browser does not support the video tag. See <a href="docs/hand_video.mp4">docs/hand_video.mp4</a>.
</video>

---

## Prerequisites

- **Python** `>=3.12, <3.15`
- **Poetry** (dependency manager) - [install instructions](https://python-poetry.org/docs/#installation)
- **ffmpeg** installed and available on `PATH` (required for MP4 export)
- **Intel RealSense `.bag` file** as input

### Hardware notes

- No GPU required - MediaPipe Hands runs on CPU
- The Intel RealSense SDK (`pyrealsense2`) is used to read `.bag` files; a physical RealSense camera is **not** required at runtime

---

## Setup

1. **Clone the repository** (or ensure this directory is inside the parent project folder).

2. **Install dependencies:**

   ```bash
   poetry install
   ```

   This installs all packages listed in [pyproject.toml](pyproject.toml), including `pyrealsense2`, `mediapipe`, `opencv-python`, `numpy`, `matplotlib`, `torch`, and `torchvision`.

3. **Verify ffmpeg is available:**

   ```bash
   ffmpeg -version
   ```

   If not installed, download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your `PATH`.

---

## Quick start

```bash
poetry run python src/main.py --bag-path /absolute/path/to/input.bag
```

This generates all outputs in the `out/` directory by default.

---

## Configuration

Configuration is loaded in this order (highest priority first):

1. CLI flags
2. Environment variables
3. JSON config file (`--config`)
4. Built-in defaults

### CLI usage

```bash
poetry run python src/main.py \
    --bag-path /absolute/path/to/input.bag \
    --output-dir out \
    --frame-stride 1 \
    --viz-frame-stride 2 \
    --viz-axis-map "x,y,z" \
    --timeout-ms 3000 \
    --fill-mode interp \
    --interpolation-max-gap 3 \
    --smoothing \
    --smoothing-alpha 0.35 \
    --smoothing-method ema \
    --savgol-window 7 \
    --savgol-polyorder 2 \
    --despike-max-speed-mps 4.0 \
    --depth-min-meters 0.08 \
    --depth-max-meters 2.5 \
    --debug-viz \
    --debug-velocity-trails \
    --debug-trail-length 20
```

### All flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bag-path` | string | *(required)* | Input `.bag` file |
| `--config` | string | | Path to JSON config file |
| `--output-dir` | string | `out` | Output directory |
| `--left-npz-name` | string | `hand_3d_left.npz` | Left-hand NPZ filename |
| `--right-npz-name` | string | `hand_3d_right.npz` | Right-hand NPZ filename |
| `--left-mp4-name` | string | `hand_3d_motion_left.mp4` | Left-hand MP4 filename |
| `--right-mp4-name` | string | `hand_3d_motion_right.mp4` | Right-hand MP4 filename |
| `--frame-stride` | int | `1` | Process every Nth frame from the bag file |
| `--viz-frame-stride` | int | `2` | Frame stride for video rendering |
| `--viz-axis-map` | string | `x,y,z` | Axis remap from camera coords (e.g. `x,z,-y`) |
| `--timeout-ms` | int | `3000` | Frame read timeout in milliseconds |
| `--fill-mode` | choice | `interp` | Missing-frame handling (`interp` or `none`) |
| `--interpolation-max-gap` | int | `3` | Max consecutive missing frames to interpolate |
| `--smoothing` / `--no-smoothing` | bool | `false` | Toggle temporal smoothing |
| `--smoothing-alpha` | float | `0.35` | EMA smoothing coefficient `[0, 1]` |
| `--smoothing-method` | choice | `ema` | Smoothing method (`ema` or `savgol`) |
| `--savgol-window` | int | `7` | Savitzky-Golay window length (odd) |
| `--savgol-polyorder` | int | `2` | Savitzky-Golay polynomial order |
| `--despike-max-speed-mps` | float | `4.0` | Velocity threshold (m/s) for spike rejection |
| `--depth-min-meters` | float | `0.08` | Minimum valid depth in meters |
| `--depth-max-meters` | float | `2.5` | Maximum valid depth in meters |
| `--debug-viz` / `--no-debug-viz` | bool | `false` | Enable side-by-side RGB + 3D debug video |
| `--left-debug-mp4-name` | string | `hand_3d_motion_left_debug.mp4` | Left debug video filename |
| `--right-debug-mp4-name` | string | `hand_3d_motion_right_debug.mp4` | Right debug video filename |
| `--debug-velocity-trails` / `--no-debug-velocity-trails` | bool | `false` | Wrist velocity trail overlay (debug only) |
| `--debug-trail-length` | int | `20` | Trail length in frames |

### Environment variables

Every flag has a corresponding environment variable prefixed with `SCALPR_`:

| Variable | Corresponds to |
|---|---|
| `SCALPR_BAG_PATH` | `--bag-path` |
| `SCALPR_OUTPUT_DIR` | `--output-dir` |
| `SCALPR_LEFT_NPZ_NAME` | `--left-npz-name` |
| `SCALPR_RIGHT_NPZ_NAME` | `--right-npz-name` |
| `SCALPR_LEFT_MP4_NAME` | `--left-mp4-name` |
| `SCALPR_RIGHT_MP4_NAME` | `--right-mp4-name` |
| `SCALPR_FRAME_STRIDE` | `--frame-stride` |
| `SCALPR_VIZ_FRAME_STRIDE` | `--viz-frame-stride` |
| `SCALPR_VIZ_AXIS_MAP` | `--viz-axis-map` |
| `SCALPR_TIMEOUT_MS` | `--timeout-ms` |
| `SCALPR_FILL_MODE` | `--fill-mode` |
| `SCALPR_INTERPOLATION_MAX_GAP` | `--interpolation-max-gap` |
| `SCALPR_SMOOTHING` | `--smoothing` |
| `SCALPR_SMOOTHING_ALPHA` | `--smoothing-alpha` |
| `SCALPR_SMOOTHING_METHOD` | `--smoothing-method` |
| `SCALPR_SAVGOL_WINDOW` | `--savgol-window` |
| `SCALPR_SAVGOL_POLYORDER` | `--savgol-polyorder` |
| `SCALPR_DESPIKE_MAX_SPEED_MPS` | `--despike-max-speed-mps` |
| `SCALPR_DEPTH_MIN_METERS` | `--depth-min-meters` |
| `SCALPR_DEPTH_MAX_METERS` | `--depth-max-meters` |
| `SCALPR_DEBUG_VIZ` | `--debug-viz` |
| `SCALPR_LEFT_DEBUG_MP4_NAME` | `--left-debug-mp4-name` |
| `SCALPR_RIGHT_DEBUG_MP4_NAME` | `--right-debug-mp4-name` |
| `SCALPR_DEBUG_VELOCITY_TRAILS` | `--debug-velocity-trails` |
| `SCALPR_DEBUG_TRAIL_LENGTH` | `--debug-trail-length` |

Example:

```bash
export SCALPR_BAG_PATH=/absolute/path/to/input.bag
export SCALPR_SMOOTHING=true
poetry run python src/main.py
```

### JSON config file

Pass a config file with `--config`:

```bash
poetry run python src/main.py --config config.json
```

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
    "viz_axis_map": "x,y,z",
    "timeout_ms": 3000,
    "fill_mode": "interp",
    "interpolation_max_gap": 3,
    "smoothing": true,
    "smoothing_alpha": 0.35,
    "smoothing_method": "ema",
    "savgol_window": 7,
    "savgol_polyorder": 2,
    "despike_max_speed_mps": 4.0,
    "depth_min_meters": 0.08,
    "depth_max_meters": 2.5,
    "debug_viz": false,
    "debug_velocity_trails": false,
    "debug_trail_length": 20
}
```

---

## Outputs

By default, `out/` contains:

| File | Description |
|---|---|
| `hand_3d_left.npz` | Left-hand 3D trajectory |
| `hand_3d_right.npz` | Right-hand 3D trajectory |
| `hand_3d_motion_left.mp4` | Left-hand skeleton animation |
| `hand_3d_motion_right.mp4` | Right-hand skeleton animation |
| `run_quality_report.json` | Per-run quality metrics |

With `--debug-viz` enabled, two additional files are written:

| File | Description |
|---|---|
| `hand_3d_motion_left_debug.mp4` | Side-by-side RGB + 3D skeleton (left) |
| `hand_3d_motion_right_debug.mp4` | Side-by-side RGB + 3D skeleton (right) |

### NPZ contents

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `times` | `(T,)` | Timestamps in seconds |
| `positions` | `(T, 21, 3)` | Final output track |
| `raw_positions` | `(T, 21, 3)` | Raw 3D joint positions from depth back-projection |
| `despiked_positions` | `(T, 21, 3)` | Positions after velocity spike rejection |
| `spike_mask` | `(T, 21)` | Boolean mask of joints flagged as spikes |
| `filled_positions` | `(T, 21, 3)` | Positions after gap interpolation |
| `smoothed_positions` | `(T, 21, 3)` | Positions after smoothing |
| `pre_post_despike_smoothed_positions` | `(T, 21, 3)` | Smoothed positions before final guard despike |
| `post_smoothing_spike_mask` | `(T, 21)` | Spike mask applied after smoothing |
| `final_guard_spike_mask` | `(T, 21)` | Final guard spike mask |
| `source_frame_stride` | scalar | Frame stride used during extraction |
| `fill_mode` | string | Gap-fill mode used (`interp` or `none`) |
| `interpolation_max_gap` | scalar | Max gap size for interpolation |
| `smoothing_enabled` | bool | Whether smoothing was applied |
| `smoothing_method` | string | Smoothing method used |
| `smoothing_alpha` | scalar | EMA alpha value |
| `savgol_window` | scalar | Savitzky-Golay window length |
| `savgol_polyorder` | scalar | Savitzky-Golay polynomial order |

### Debug visualisation

When `--debug-viz` is enabled, debug videos show:
- Side-by-side RGB frame and 3D skeleton
- Consistent left (red) / right (blue) colour scheme
- Interpolated joints marked with `x`
- Optional wrist velocity trail (`--debug-velocity-trails`)

Velocity trails are debug-only; standard output videos do not include them.

---

## Quality report

After extraction, `src/main.py` automatically runs `src/report.py` to generate `run_quality_report.json`.

Reported metrics (per hand):

| Metric | Warning threshold |
|---|---|
| Hand detection rate | < 90% |
| Valid-joint percentage | < 90% |
| Longest missing streak | > 15 frames |
| Left/right frame count mismatch | > 0 |
| Left/right detected-frame mismatch | > 30 |

The report sets `status` to `"ok"` or `"warn"` based on these thresholds.

You can also run the report standalone with custom thresholds:

```bash
poetry run python src/report.py \
    --left-npz out/hand_3d_left.npz \
    --right-npz out/hand_3d_right.npz \
    --report-out out/run_quality_report.json \
    --min-detection-rate 0.90 \
    --min-valid-joint-percentage 0.90 \
    --max-missing-streak 15 \
    --max-frame-count-mismatch 0 \
    --max-detected-frame-mismatch 30
```

---

## Testing

Run the test suite:

```bash
poetry run python -m unittest discover -s tests -v
```

Tests included:

- **`test_bagtohands.py`** - unit tests for the core extraction logic:
  - `_get_depth_at`: median depth from a valid patch, NaN when no valid depth
  - `_landmarks_to_3d`: 3D projection of landmarks with NaN propagation for out-of-bounds joints
  - `_compute_spike_mask`: velocity-based spike detection flags implausible jumps
- **`test_motionv_smoke.py`** - visualiser tests:
  - Axis map transform: verifies axis reordering and flipping
  - MP4 smoke test: writes a video from a tiny synthetic NPZ (skipped if ffmpeg is not available)

---

## Project structure

```
src/
    main.py          # End-to-end orchestration (extract, report, visualise)
    bagtohands.py    # Core pipeline: RealSense + MediaPipe + post-processing
    motionv.py       # 3D skeleton renderer and debug visualisation
    config.py        # CLI / env / JSON configuration loading
    report.py        # Quality metrics and threshold-based warnings
scripts/
    diagnose_spikes.py   # Diagnostic script for investigating velocity spikes
tests/
    test_bagtohands.py   # Unit tests for extraction logic
    test_motionv_smoke.py # Visualiser unit + smoke tests
pyproject.toml       # Project metadata and dependencies
```
