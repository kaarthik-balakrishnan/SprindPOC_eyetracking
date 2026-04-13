#!/usr/bin/env python3
"""
Step 2: Eye Centering & Contrast Enhancement

- Crops and centers the eye region in each frame
- Applies CLAHE contrast enhancement
- Uses optical flow to track eye center between frames

Usage:
  python scripts/eye_center.py --input outputs/stabilized.mp4 \\
    --output outputs/eye_centered.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _print_progress(current, total, prefix="Progress:", bar_length=40):
    """Simple text-based progress bar."""
    if total <= 0:
        return
    percent = current / total
    filled = int(bar_length * percent)
    bar = "=" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _clahe_enhance(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def find_eye_center(gray: np.ndarray) -> tuple[int, int]:
    """Find approximate eye center using template matching or heuristics."""
    h, w = gray.shape
    
    # Use moments to find approximate center of mass
    # Works well for iris/pupil which are darker than surroundings
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    moments = cv2.moments(thresh)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    
    return w // 2, h // 2


def center_eye_frame(
    frame: np.ndarray,
    eye_x: int,
    eye_y: int,
    crop_size: int = 200,
) -> tuple[np.ndarray, int, int]:
    """Crop frame around eye center, centering the eye."""
    h, w = frame.shape[:2]
    
    # Calculate crop bounds centered on eye
    x1 = max(0, eye_x - crop_size // 2)
    x2 = min(w, eye_x + crop_size // 2)
    y1 = max(0, eye_y - crop_size // 2)
    y2 = min(h, eye_y + crop_size // 2)
    
    cropped = frame[y1:y2, x1:x2]
    
    # Pad if eye is too close to edge
    pad_x = max(0, crop_size - cropped.shape[1])
    pad_y = max(0, crop_size - cropped.shape[0])
    
    if pad_x > 0 or pad_y > 0:
        cropped = cv2.copyMakeBorder(
            cropped, 
            0, pad_y, 
            0, pad_x, 
            cv2.BORDER_CONSTANT, 
            value=(0, 0, 0)
        )
    
    return cropped, x1, y1


def main() -> int:
    parser = argparse.ArgumentParser(description="Eye centering and contrast enhancement (Step 2).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=200,
        help="Size of cropped eye region (pixels).",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames.",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=5,
        help="Optical flow smoothing radius for eye tracking.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Failed to open: {args.input}", file=sys.stderr)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Read first frame
    ok, frame0 = cap.read()
    if not ok:
        print("Could not read first frame.", file=sys.stderr)
        cap.release()
        return 1

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    eye_x, eye_y = find_eye_center(gray0)
    
    # Apply CLAHE to first frame
    gray0_enhanced = _clahe_enhance(gray0, clip_limit=args.clip_limit)
    
    # Center eye in first frame
    first_cropped = frame0.copy()
    if len(frame0.shape) == 3:
        # Process color: enhance and center
        cropped_gray = cv2.cvtColor(first_cropped, cv2.COLOR_BGR2GRAY)
        cropped_enhanced = _clahe_enhance(cropped_gray, clip_limit=args.clip_limit)
        # Apply enhanced as separate channel or convert back
        # For simplicity, enhance grayscale and use it
        first_cropped = cv2.cvtColor(cropped_enhanced, cv2.COLOR_GRAY2BGR)
    
    first_cropped, _, _ = center_eye_frame(first_cropped, eye_x, eye_y, args.crop_size)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_h, out_w = first_cropped.shape[:2]
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.output}", file=sys.stderr)
        cap.release()
        return 1

    writer.write(first_cropped)
    
    # Track eye positions
    eye_positions = [(eye_x, eye_y)]
    prev_gray = gray0_enhanced
    prev_crop = first_cropped.copy()
    
    frame_count = 1
    max_frames = args.max_frames if args.max_frames else total_frames
    
    print(f"Processing {min(max_frames, total_frames)} frames...")
    
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        
        _print_progress(frame_count + 1, max_frames, "  ")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_enhanced = _clahe_enhance(gray, clip_limit=args.clip_limit)
        
        # Track eye with optical flow
        prev_pts = np.array([[eye_x, eye_y]], dtype=np.float32).reshape(1, 1, 2)
        
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray_enhanced,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        
        if next_pts is not None and status[0][0]:
            eye_x, eye_y = int(next_pts[0][0][0]), int(next_pts[0][0][1])
        
        # Smooth eye position
        if len(eye_positions) >= args.smooth_radius * 2 + 1:
            window = eye_positions[-args.smooth_radius * 2 - 1:]
            eye_x = int(np.mean([p[0] for p in window]))
            eye_y = int(np.mean([p[1] for p in window]))
        
        eye_positions.append((eye_x, eye_y))
        
        # Crop and center
        cropped, _, _ = center_eye_frame(frame, eye_x, eye_y, args.crop_size)
        
        # Apply CLAHE to cropped region
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped_enhanced = _clahe_enhance(cropped_gray, clip_limit=args.clip_limit)
        cropped_final = cv2.cvtColor(cropped_enhanced, cv2.COLOR_GRAY2BGR)
        
        writer.write(cropped_final)
        
        prev_gray = gray_enhanced
        frame_count += 1

    cap.release()
    writer.release()

    print(f"\nProcessed {frame_count} frames.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
