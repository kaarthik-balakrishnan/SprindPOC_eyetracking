#!/usr/bin/env python3
"""
Step 1: Stabilize video using averaged reference frame

- Takes average of first 5 frames as reference
- Stabilizes all frames to match reference using feature matching
- Supports rotation, translation, and scale (zoom)

Usage:
  python scripts/stabilize_video.py --input data/video.mp4 \\
    --output outputs/stabilized.mp4
"""

from __future__ import annotations

import argparse
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


def create_reference_frame(cap: cv2.VideoCapture, num_frames: int = 5) -> np.ndarray:
    """Create reference frame by averaging first N frames."""
    frames = []
    
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.astype(np.float32))
    
    if frames:
        reference = np.mean(frames, axis=0).astype(np.uint8)
    else:
        ok, reference = cap.read()
        if not ok:
            raise RuntimeError("Could not read any frames")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return reference


def find_transform(
    reference_gray: np.ndarray,
    frame_gray: np.ndarray,
    max_corners: int = 200,
    quality: float = 0.01,
    min_distance: int = 30,
) -> tuple[np.ndarray | None, float]:
    """
    Find affine/similarity transform between reference and frame.
    Returns transform matrix and scale factor.
    """
    ref_pts = cv2.goodFeaturesToTrack(
        reference_gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
    )
    
    if ref_pts is None or len(ref_pts) < 4:
        return None, 1.0
    
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        frame_gray,
        ref_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    
    if next_pts is None:
        return None, 1.0
    
    status = status.ravel().astype(bool)
    ref_good = ref_pts[status]
    next_good = next_pts[status]
    
    if len(ref_good) < 4:
        return None, 1.0
    
    M, inliers = cv2.estimateAffinePartial2D(ref_good, next_good, method=cv2.RANSAC)
    
    if M is None:
        return None, 1.0
    
    return M, 1.0


def smooth_transforms(transforms: list[np.ndarray], radius: int = 15) -> list[np.ndarray]:
    """Smooth transforms using moving average."""
    if radius <= 0 or len(transforms) <= radius * 2:
        return transforms
    
    smoothed = []
    half = radius
    
    for i in range(len(transforms)):
        start = max(0, i - half)
        end = min(len(transforms), i + half + 1)
        window = transforms[start:end]
        
        avg_transform = np.mean(window, axis=0)
        smoothed.append(avg_transform)
    
    return smoothed


def apply_transform(
    frame: np.ndarray,
    M: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Apply transform to frame and crop to target size."""
    stabilized = cv2.warpAffine(
        frame,
        M,
        target_size,
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return stabilized


def main() -> int:
    parser = argparse.ArgumentParser(description="Stabilize video using reference frame (Step 1).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument(
        "--ref-frames",
        type=int,
        default=5,
        help="Number of frames to average for reference.",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=15,
        help="Smoothing radius for transforms.",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Pixels to crop on each side.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for processing.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Failed to open: {args.input}", file=sys.stderr)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    w, h = orig_w, orig_h
    if args.scale != 1.0:
        w, h = int(w * args.scale), int(h * args.scale)

    print(f"Creating reference frame from first {args.ref_frames} frames...")
    reference = create_reference_frame(cap, args.ref_frames)
    
    if args.scale != 1.0:
        reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
    
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    out_w = w - 2 * args.crop_border
    out_h = h - 2 * args.crop_border
    
    if out_w <= 0 or out_h <= 0:
        print("crop_border too large.", file=sys.stderr)
        cap.release()
        return 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.output}", file=sys.stderr)
        cap.release()
        return 1

    print(f"Pass 1/2: Computing transforms ({total_frames} frames)...")
    
    transforms = []
    max_frames = args.max_frames if args.max_frames else total_frames
    
    for frame_idx in range(max_frames):
        _print_progress(frame_idx + 1, max_frames, "  ")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        if args.scale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_idx < args.ref_frames:
            transforms.append(np.eye(2, 3, dtype=np.float64))
            continue
        
        M, _ = find_transform(ref_gray, gray)
        
        if M is not None:
            transforms.append(M)
        else:
            transforms.append(np.eye(2, 3, dtype=np.float64))

    print(f"\nPass 2/2: Applying transforms...")
    
    smoothed_transforms = smooth_transforms(transforms, args.smooth_radius)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    identity = np.eye(2, 3, dtype=np.float64)
    
    for frame_idx in range(max_frames):
        _print_progress(frame_idx + 1, max_frames, "  ")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        if args.scale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        M = smoothed_transforms[frame_idx] if frame_idx < len(smoothed_transforms) else identity
        
        stabilized = apply_transform(frame, M, (w, h))
        
        if args.crop_border > 0:
            stabilized = stabilized[
                args.crop_border:h - args.crop_border,
                args.crop_border:w - args.crop_border
            ]
        
        writer.write(stabilized)

    cap.release()
    writer.release()

    print(f"\nProcessed {max_frames} frames.")
    print(f"Output: {args.output}")
    
    import subprocess
    temp_output = str(args.output) + ".tmp.mp4"
    try:
        print("Re-encoding with H.264 compression...")
        subprocess.run([
            'ffmpeg', '-y', '-i', str(args.output),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k',
            temp_output
        ], capture_output=True, check=True)
        import shutil
        shutil.move(temp_output, args.output)
        print(f"Compressed: {args.output}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Uncompressed output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
