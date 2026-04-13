# Eye Tracking Pipeline

A computer vision pipeline for tracking eye movements from handheld video footage, estimating gaze direction in 3D space.

## Pipeline Overview

| Step | Description | Output |
|------|-------------|--------|
| [1. Stabilization](docs/step1_stabilization.md) | Remove camera shake using reference frame alignment | Stabilized video |
| [2. Eye Centering](docs/step2_eye_centering.md) | Crop and center eye region, apply CLAHE | Eye-centered video |
| [3. Pupil Tracking](docs/step3_pupil_tracking.md) | Detect pupil using HoughCircles | CSV with (x, y, radius) |
| [4. 3D Gaze Estimation](docs/step4_gaze_3d.md) | Fit sphere, compute gaze vectors | CSV with 3D vectors |

## Quick Start

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaarthik-balakrishnan/SprindPOC_eyetracking/blob/main/notebooks/eye_tracking_pipeline.ipynb)

1. Open the notebook in Colab
2. Upload your video to Google Drive
3. Update paths in the notebook
4. Run all cells sequentially

### Local

```bash
pip install -r requirements.txt

# Step 1: Stabilize
python scripts/stabilize_video.py --input video.mp4 --output outputs/stabilized.mp4

# Step 2: Eye centering
python scripts/eye_center.py --input outputs/stabilized.mp4 --output outputs/eye_centered.mp4

# Step 3: Pupil tracking
python scripts/track_pupil.py --input outputs/eye_centered.mp4 --output outputs/pupil_data.csv

# Step 4: 3D gaze estimation
python scripts/gaze_3d.py --input outputs/pupil_data.csv --output outputs/gaze_3d.csv
```

## Project Structure

```
.
├── scripts/
│   ├── stabilize_video.py   # Step 1: Video stabilization
│   ├── eye_center.py        # Step 2: Eye centering & CLAHE
│   ├── track_pupil.py       # Step 3: Pupil detection
│   └── gaze_3d.py           # Step 4: 3D sphere fitting
├── docs/
│   ├── step1_stabilization.md
│   ├── step2_eye_centering.md
│   ├── step3_pupil_tracking.md
│   └── step4_gaze_3d.md
├── notebooks/
│   └── eye_tracking_pipeline.ipynb
├── outputs/                  # Generated files (not tracked)
├── OPTIMIZATIONS.md          # Optimization recommendations
└── README.md
```

## Output Files

| File | Format | Contents |
|------|--------|----------|
| `stabilized.mp4` | Video | Camera-stabilized footage |
| `eye_centered.mp4` | Video | Eye region cropped and centered |
| `pupil_data.csv` | CSV | Frame-by-frame pupil position and radius |
| `gaze_3d.csv` | CSV | 3D gaze direction vectors |
| `gaze_plots.png` | Image | Visualization of gaze data |

## Optimization Status

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for planned optimizations and current status.

## Requirements

- Python 3.8+
- OpenCV (opencv-python-headless)
- NumPy
- Pandas (for visualization)
- Matplotlib (for visualization)

## License

MIT
