#!/usr/bin/env python3
"""
Step 3: Pupil Tracking using Circle Detection (HoughCircles)

- Detects pupil as circle using Hough Transform
- Tracks pupil position across frames
- Outputs pupil coordinates and radius for each frame

Usage:
  python scripts/track_pupil.py --input outputs/eye_centered.mp4 \\
    --output outputs/pupil_data.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def _print_progress(current, total, prefix="Progress:", bar_length=40):
    if total <= 0:
        return
    percent = current / total
    filled = int(bar_length * percent)
    bar = "=" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def preprocess_for_pupil(gray: np.ndarray) -> np.ndarray:
    """Preprocess grayscale image for pupil detection."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return blurred


def detect_pupil(
    gray: np.ndarray,
    min_radius: int = 10,
    max_radius: int = 80,
    dp: float = 1.0,
    param1: float = 50.0,
    param2: float = 30.0,
) -> tuple[int, int, int] | None:
    """Detect pupil using HoughCircles."""
    processed = preprocess_for_pupil(gray)
    
    circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_radius,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        best = circles[0][0]
        return int(best[0]), int(best[1]), int(best[2])
    
    return None


def detect_pupil_blob(gray: np.ndarray) -> tuple[int, int, int] | None:
    """Fallback: Detect pupil using blob detector."""
    processed = preprocess_for_pupil(gray)
    
    _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    
    if radius > 5:
        return int(x), int(y), int(radius)
    
    return None


def track_pupil_optical_flow(
    gray: np.ndarray,
    prev_x: int,
    prev_y: int,
    prev_radius: int,
) -> tuple[int, int, int]:
    """Track pupil using optical flow from previous position."""
    prev_pts = np.array([[[prev_x, prev_y]]], dtype=np.float32)
    
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    
    if next_pts is not None and status[0][0]:
        return int(next_pts[0][0][0]), int(next_pts[0][0][1]), prev_radius
    
    return prev_x, prev_y, prev_radius


def main() -> int:
    parser = argparse.ArgumentParser(description="Track pupil position (Step 3).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--min-radius",
        type=int,
        default=10,
        help="Minimum pupil radius (pixels).",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=80,
        help="Maximum pupil radius (pixels).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Failed to open: {args.input}", file=sys.stderr)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_num = 0
    pupil_data = []
    prev_x, prev_y, prev_radius = 0, 0, 0
    has_prev = False

    max_frames = args.max_frames if args.max_frames else total_frames
    
    print(f"Tracking pupil in {min(max_frames, total_frames)} frames...")
    print(f"Processing every {args.skip_frames} frame(s)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_num >= max_frames:
            break

        if frame_num % args.skip_frames != 0:
            frame_num += 1
            continue

        _print_progress(frame_num + 1, max_frames, "  ")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if has_prev:
            prev_x, prev_y, prev_radius = track_pupil_optical_flow(
                gray, prev_x, prev_y, prev_radius
            )
        
        pupil = detect_pupil(
            gray,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
        )
        
        if pupil is None and has_prev:
            pupil = detect_pupil_blob(gray)
        
        if pupil is not None:
            x, y, radius = pupil
            prev_x, prev_y, prev_radius = x, y, radius
            has_prev = True
        elif has_prev:
            x, y, radius = prev_x, prev_y, prev_radius
        else:
            x, y, radius = -1, -1, -1

        pupil_data.append({
            'frame': frame_num,
            'x': x,
            'y': y,
            'radius': radius,
        })

        frame_num += 1

    cap.release()

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'x', 'y', 'radius'])
        writer.writeheader()
        writer.writerows(pupil_data)

    detected = sum(1 for p in pupil_data if p['x'] >= 0)
    print(f"\nProcessed {len(pupil_data)} frames.")
    print(f"Detected pupil in {detected}/{len(pupil_data)} frames ({100*detected/len(pupil_data):.1f}%)")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
