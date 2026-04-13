#!/usr/bin/env python3
"""
Step 2: Eye Tracking with Circle Overlay

- Applies CLAHE contrast enhancement to make pupil visible
- Detects pupil using dark region analysis
- Tracks pupil using multi-point optical flow (handles saccades)
- Draws white circle around pupil

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


def enhance_contrast(gray: np.ndarray, clip_limit: float = 4.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE for better pupil visibility."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def detect_pupil_center(gray: np.ndarray) -> tuple[int, int, int]:
    """
    Detect pupil center using dark region analysis and circle fitting.
    Returns (x, y, radius).
    """
    h, w = gray.shape
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return w // 2, h // 2, 20
    
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(areas)
    contour = contours[max_idx]
    
    area = areas[max_idx]
    if area < 10:
        return w // 2, h // 2, 20
    
    (x, y), radius = cv2.minEnclosingCircle(contour)
    
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy, int(radius)
    
    return int(x), int(y), int(radius)


def create_pupil_tracking_points(gray: np.ndarray, cx: int, cy: int, radius: int, num_points: int = 8) -> np.ndarray:
    """Create tracking points around the pupil perimeter."""
    points = []
    angle_step = 2 * np.pi / num_points
    
    for i in range(num_points):
        angle = i * angle_step
        px = int(cx + radius * np.cos(angle))
        py = int(cy + radius * np.sin(angle))
        
        h, w = gray.shape
        if 0 <= px < w and 0 <= py < h:
            points.append([px, py])
        
        px2 = int(cx + (radius + 5) * np.cos(angle))
        py2 = int(cy + (radius + 5) * np.sin(angle))
        if 0 <= px2 < w and 0 <= py2 < h:
            points.append([px2, py2])
    
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)


def track_pupil_multi_point(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    Track multiple points using optical flow.
    Uses RANSAC to filter outliers.
    Returns (tracked_points, status, errors).
    """
    next_pts, status, errors = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    
    if next_pts is None or status is None:
        return None, None, None
    
    status = status.ravel().astype(bool)
    next_pts = next_pts.reshape(-1, 2)
    prev_pts_reshaped = prev_pts.reshape(-1, 2)
    
    valid_next = next_pts[status]
    valid_prev = prev_pts_reshaped[status]
    valid_errors = errors.ravel()[status] if errors is not None else np.array([])
    
    if len(valid_next) < 4:
        return valid_prev.reshape(-1, 1, 2) if len(valid_next) > 0 else None, None, valid_errors
    
    try:
        M, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_next, method=cv2.RANSAC)
        if M is not None and inliers is not None:
            inlier_mask = inliers.ravel().astype(bool)
            valid_next = valid_next[inlier_mask]
            valid_prev = valid_prev[inlier_mask]
    except:
        pass
    
    return valid_prev.reshape(-1, 1, 2) if len(valid_next) > 0 else None, valid_next.reshape(-1, 1, 2) if len(valid_next) > 0 else None, valid_errors


def compute_pupil_from_points(tracked_pts: np.ndarray, ref_cx: int, ref_cy: int) -> tuple[int, int, int]:
    """Compute pupil center from tracked points using RANSAC circle fitting."""
    if tracked_pts is None or len(tracked_pts) < 3:
        return ref_cx, ref_cy, 20
    
    pts = tracked_pts.reshape(-1, 2)
    
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    
    distances = np.sqrt((pts[:, 0] - center_x)**2 + (pts[:, 1] - center_y)**2)
    median_radius = np.median(distances)
    
    inlier_threshold = median_radius * 1.5
    inliers = distances < inlier_threshold
    
    if np.sum(inliers) >= 3:
        inlier_pts = pts[inliers]
        center_x = np.mean(inlier_pts[:, 0])
        center_y = np.mean(inlier_pts[:, 1])
        median_radius = np.median(np.sqrt((inlier_pts[:, 0] - center_x)**2 + (inlier_pts[:, 1] - center_y)**2))
    
    return int(center_x), int(center_y), max(int(median_radius), 10)


def smooth_positions(positions: list, radius: int = 5) -> list:
    """Smooth pupil positions using moving average."""
    if len(positions) < radius * 2 + 1:
        return positions
    
    smoothed = []
    half = radius
    
    for i in range(len(positions)):
        start = max(0, i - half)
        end = min(len(positions), i + half + 1)
        window = positions[start:end]
        
        avg_x = np.mean([p[0] for p in window])
        avg_y = np.mean([p[1] for p in window])
        avg_r = np.mean([p[2] for p in window])
        
        smoothed.append((avg_x, avg_y, avg_r))
    
    return smoothed


def draw_pupil_circle(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw circle around pupil with crosshairs."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    cx = int(cx)
    cy = int(cy)
    radius = int(radius)
    
    if 0 <= cx < w and 0 <= cy < h:
        cv2.circle(output, (cx, cy), radius, color, thickness)
        
        cv2.line(output, (cx - radius - 15, cy), (cx - radius - 5, cy), color, 1)
        cv2.line(output, (cx + radius + 5, cy), (cx + radius + 15, cy), color, 1)
        cv2.line(output, (cx, cy - radius - 15), (cx, cy - radius - 5), color, 1)
        cv2.line(output, (cx, cy + radius + 5), (cx, cy + radius + 15), color, 1)
        
        cv2.circle(output, (cx, cy), 3, color, -1)
    
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Track pupil and draw circle overlay (Step 2).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=4.0,
        help="CLAHE clip limit (higher = more contrast).",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=5,
        help="Smoothing radius for positions.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=12,
        help="Number of tracking points around pupil.",
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=25,
        help="Base radius of circle to draw.",
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.output}", file=sys.stderr)
        cap.release()
        return 1

    max_frames = args.max_frames if args.max_frames else total_frames
    
    print(f"Processing {max_frames} frames...")
    
    pupil_positions = []
    prev_pts = None
    prev_enhanced = None
    first_frame = True
    
    for frame_idx in range(max_frames):
        _print_progress(frame_idx + 1, max_frames, "  ")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        if args.scale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = enhance_contrast(gray, clip_limit=args.clip_limit)
        
        if first_frame:
            cx, cy, radius = detect_pupil_center(enhanced)
            prev_pts = create_pupil_tracking_points(enhanced, cx, cy, radius, args.num_points)
            prev_enhanced = enhanced.copy()
            first_frame = False
            
            pupil_positions.append((cx, cy, radius))
            frame_with_circle = draw_pupil_circle(frame, cx, cy, args.circle_radius)
            writer.write(frame_with_circle)
            frame_idx += 1
            continue
        
        tracked_pts, next_pts, _ = track_pupil_multi_point(prev_enhanced, enhanced, prev_pts)
        
        if tracked_pts is not None and len(tracked_pts) >= 3:
            cx, cy, detected_radius = compute_pupil_from_points(
                next_pts if next_pts is not None else tracked_pts,
                pupil_positions[-1][0],
                pupil_positions[-1][1]
            )
            
            prev_pts = create_pupil_tracking_points(enhanced, cx, cy, detected_radius, args.num_points)
        else:
            cx, cy, detected_radius = detect_pupil_center(enhanced)
            prev_pts = create_pupil_tracking_points(enhanced, cx, cy, detected_radius, args.num_points)
        
        pupil_positions.append((cx, cy, detected_radius))
        prev_enhanced = enhanced.copy()
        
        frame_with_circle = draw_pupil_circle(frame, cx, cy, args.circle_radius)
        writer.write(frame_with_circle)

    cap.release()
    writer.release()
    
    if len(pupil_positions) > args.smooth_radius * 2:
        print(f"\nSmoothing positions...")
        smoothed = smooth_positions(pupil_positions, args.smooth_radius)
        
        print("Rewriting with smoothed positions...")
        
        cap = cv2.VideoCapture(str(args.input))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
        
        for frame_idx in range(min(max_frames, len(smoothed))):
            ok, frame = cap.read()
            if not ok:
                break
            
            if args.scale != 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            
            cx, cy, _ = smoothed[frame_idx]
            frame_with_circle = draw_pupil_circle(frame, cx, cy, args.circle_radius)
            writer.write(frame_with_circle)
        
        cap.release()
        writer.release()

    print(f"\nProcessed {len(pupil_positions)} frames.")

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
