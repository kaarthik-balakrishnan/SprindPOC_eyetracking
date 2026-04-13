#!/usr/bin/env python3
"""
Step 2: Eye Tracking with Circle Overlay

- Tracks eye position across frames using feature-based alignment to reference
- Detects pupil/iris position
- Draws white circle around the eye center
- Applies stabilization transform to keep overlay stable

Usage:
  python scripts/eye_center.py --input outputs/stabilized.mp4 \\
    --output outputs/eye_tracked.mp4
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


def create_reference_frame(cap: cv2.VideoCapture, num_frames: int = 5) -> tuple[np.ndarray, int, int]:
    """Create reference frame by averaging first N frames, return eye center too."""
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
    
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    eye_x, eye_y = detect_eye_center(gray)
    
    return reference, eye_x, eye_y


def detect_eye_center(gray: np.ndarray) -> tuple[int, int]:
    """Detect eye/pupil center using multiple methods."""
    h, w = gray.shape
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=80,
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        best = circles[0][0]
        return int(best[0]), int(best[1])
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        
        (x, y), radius = cv2.minEnclosingCircle(largest)
        return int(x), int(y)
    
    return w // 2, h // 2


def find_transform_to_reference(
    ref_gray: np.ndarray,
    frame_gray: np.ndarray,
    ref_eye_x: int,
    ref_eye_y: int,
    crop_size: int = 200,
    max_corners: int = 100,
) -> tuple[np.ndarray | None, float]:
    """
    Find transform to align frame to reference, focusing on region around eye.
    Returns transform matrix and scale factor.
    """
    h, w = ref_gray.shape
    
    x1 = max(0, ref_eye_x - crop_size)
    x2 = min(w, ref_eye_x + crop_size)
    y1 = max(0, ref_eye_y - crop_size)
    y2 = min(h, ref_eye_y + crop_size)
    
    ref_crop = ref_gray[y1:y2, x1:x2]
    
    ref_pts = cv2.goodFeaturesToTrack(
        ref_crop,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
    )
    
    if ref_pts is None or len(ref_pts) < 4:
        ref_pts = cv2.goodFeaturesToTrack(
            ref_gray,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=30,
        )
        if ref_pts is None:
            return None, 1.0
    else:
        ref_pts[:, 0, 0] += x1
        ref_pts[:, 0, 1] += y1
    
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        ref_gray,
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
    
    M, _ = cv2.estimateAffinePartial2D(ref_good, next_good, method=cv2.RANSAC)
    
    return M, 1.0


def smooth_transforms(transforms: list, radius: int = 15) -> list:
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


def draw_eye_circle(
    frame: np.ndarray,
    eye_x: int,
    eye_y: int,
    radius: int = 40,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a white circle around the eye center."""
    output = frame.copy()
    
    h, w = frame.shape[:2]
    
    if 0 <= eye_x < w and 0 <= eye_y < h:
        cv2.circle(output, (eye_x, eye_y), radius, color, thickness)
        
        cv2.line(output, (eye_x - radius - 10, eye_y), (eye_x - radius - 5, eye_y), color, 1)
        cv2.line(output, (eye_x + radius + 5, eye_y), (eye_x + radius + 10, eye_y), color, 1)
        cv2.line(output, (eye_x, eye_y - radius - 10), (eye_x, eye_y - radius - 5), color, 1)
        cv2.line(output, (eye_x, eye_y + radius + 5), (eye_x, eye_y + radius + 10), color, 1)
    
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Track eye and draw circle overlay (Step 2).")
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
        "--crop-size",
        type=int,
        default=200,
        help="Search region size around eye.",
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=40,
        help="Radius of circle to draw around eye.",
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
    reference, ref_eye_x, ref_eye_y = create_reference_frame(cap, args.ref_frames)
    
    if args.scale != 1.0:
        reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
        ref_eye_x, ref_eye_y = int(ref_eye_x * args.scale), int(ref_eye_y * args.scale)
    
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.output}", file=sys.stderr)
        cap.release()
        return 1

    print(f"Pass 1/2: Computing transforms ({total_frames} frames)...")
    
    transforms = []
    max_frames = args.max_frames if args.max_frames else total_frames
    
    eye_x, eye_y = ref_eye_x, ref_eye_y
    
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
        else:
            M, _ = find_transform_to_reference(
                ref_gray, gray, ref_eye_x, ref_eye_y,
                crop_size=args.crop_size
            )
            
            if M is not None:
                transforms.append(M)
            else:
                transforms.append(np.eye(2, 3, dtype=np.float64))
            
            eye_x, eye_y = detect_eye_center(gray)
    
    print(f"\nPass 2/2: Applying transforms and drawing circles...")
    
    smoothed_transforms = smooth_transforms(transforms, args.smooth_radius)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    eye_x, eye_y = ref_eye_x, ref_eye_y
    identity = np.eye(2, 3, dtype=np.float64)
    
    for frame_idx in range(max_frames):
        _print_progress(frame_idx + 1, max_frames, "  ")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        if args.scale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        M = smoothed_transforms[frame_idx] if frame_idx < len(smoothed_transforms) else identity
        
        aligned = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        frame_with_circle = draw_eye_circle(
            aligned, ref_eye_x, ref_eye_y,
            radius=args.circle_radius,
            color=(255, 255, 255),
            thickness=2
        )
        
        writer.write(frame_with_circle)

    cap.release()
    writer.release()

    print(f"\nProcessed {max_frames} frames.")
    
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
        print(f"Output: {args.output} (uncompressed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
