#!/usr/bin/env python3
"""
Step 2: Eye Tracking with Ellipse Overlay

- Converts to grayscale and inverts to make pupil bright
- Uses ellipse fitting to detect iris (handles sideways gazes)
- Tracks using multi-point optical flow
- Draws ellipse on original video

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


def preprocess_for_eye_detection(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess frame for eye detection:
    1. Convert to grayscale
    2. Apply CLAHE for contrast enhancement
    3. Invert colors (eye becomes bright white)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    inverted = cv2.bitwise_not(enhanced)
    
    return gray, inverted


def detect_iris_ellipse(inverted: np.ndarray, frame_center: tuple[int, int], 
                       min_axis: int = 8, max_axis: int = 60) -> tuple[int, int, int, int, float] | None:
    """
    Detect iris using ellipse fitting.
    Returns (cx, cy, a, b, angle) or None if no ellipse found.
    """
    h, w = inverted.shape
    
    _, thresh = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_ellipse = None
    best_score = float('inf')
    
    for cnt in contours:
        if len(cnt) < 5:
            continue
        
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        
        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue
        
        (cx, cy), (a, b), angle = ellipse
        
        if a < min_axis or b < min_axis:
            continue
        if a > max_axis or b > max_axis:
            continue
        
        aspect_ratio = min(a, b) / max(a, b)
        if aspect_ratio < 0.4:
            continue
        
        dist = np.sqrt((cx - frame_center[0])**2 + (cy - frame_center[1])**2)
        area_score = abs(area - np.pi * a * b) / (np.pi * a * b)
        
        score = dist + area_score * 100
        
        if score < best_score:
            best_score = score
            best_ellipse = ellipse
    
    if best_ellipse is None:
        return None
    
    (cx, cy), (a, b), angle = best_ellipse
    
    return int(cx), int(cy), int(a), int(b), angle


def detect_iris_circle(inverted: np.ndarray, frame_center: tuple[int, int],
                       min_radius: int = 8, max_radius: int = 60) -> tuple[int, int, int] | None:
    """
    Fallback: Detect iris using contour moments (circle approximation).
    Returns (cx, cy, radius) or None.
    """
    _, thresh = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_contour = None
    best_score = float('inf')
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        if radius < min_radius or radius > max_radius:
            continue
        
        dist = np.sqrt((x - frame_center[0])**2 + (y - frame_center[1])**2)
        
        if dist < best_score:
            best_score = dist
            best_contour = cnt
    
    if best_contour is None:
        return None
    
    M = cv2.moments(best_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        (x, y), r = cv2.minEnclosingCircle(best_contour)
        return cx, cy, int(r)
    
    return None


def create_tracking_points(cx: int, cy: int, a: int, b: int, angle: float, 
                          num_points: int = 12) -> np.ndarray:
    """
    Create tracking points along elliptical perimeter.
    Angle is in degrees, converted to radians for calculation.
    """
    points = []
    angle_rad = np.radians(angle)
    
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        px = cx + a * cos_t * cos_a - b * sin_t * sin_a
        py = cy + a * cos_t * sin_a + b * sin_t * cos_a
        
        points.append([int(px), int(py)])
        
        px2 = cx + (a * 0.6) * cos_t * cos_a - (b * 0.6) * sin_t * sin_a
        py2 = cy + (a * 0.6) * cos_t * sin_a + (b * 0.6) * sin_t * cos_a
        
        points.append([int(px2), int(py2)])
    
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)


def track_with_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Track points using Lucas-Kanade optical flow."""
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    
    if next_pts is None or status is None:
        return None, None
    
    valid_prev = prev_pts[status.ravel().astype(bool)]
    valid_next = next_pts[status.ravel().astype(bool)]
    
    if len(valid_next) < 5:
        return valid_prev.reshape(-1, 1, 2) if len(valid_prev) > 0 else None, None
    
    try:
        M, inliers = cv2.estimateAffinePartial2D(
            valid_prev.reshape(-1, 2),
            valid_next.reshape(-1, 2),
            method=cv2.RANSAC
        )
        if M is not None and inliers is not None:
            inlier_mask = inliers.ravel().astype(bool)
            valid_prev = valid_prev.reshape(-1, 2)[inlier_mask]
            valid_next = valid_next.reshape(-1, 2)[inlier_mask]
    except:
        pass
    
    return (valid_prev.reshape(-1, 1, 2) if len(valid_prev) > 0 else None,
            valid_next.reshape(-1, 1, 2) if len(valid_next) > 0 else None)


def fit_ellipse_to_points(pts: np.ndarray) -> tuple[int, int, int, int, float]:
    """
    Fit ellipse to tracked points.
    Returns (cx, cy, a, b, angle).
    """
    if pts is None or len(pts) < 5:
        return 0, 0, 25, 25, 0
    
    pts = pts.reshape(-1, 2).astype(np.float64)
    
    try:
        ellipse = cv2.fitEllipse(pts)
        (cx, cy), (a, b), angle = ellipse
        return int(cx), int(cy), int(a), int(b), angle
    except:
        center_x = np.mean(pts[:, 0])
        center_y = np.mean(pts[:, 1])
        return int(center_x), int(center_y), 25, 25, 0


def smooth_ellipses(ellipses: list, radius: int = 5) -> list:
    """Smooth ellipse parameters using moving average."""
    if len(ellipses) < radius * 2 + 1:
        return ellipses
    
    smoothed = []
    half = radius
    
    for i in range(len(ellipses)):
        start = max(0, i - half)
        end = min(len(ellipses), i + half + 1)
        window = ellipses[start:end]
        
        avg_cx = np.mean([e[0] for e in window])
        avg_cy = np.mean([e[1] for e in window])
        avg_a = np.mean([e[2] for e in window])
        avg_b = np.mean([e[3] for e in window])
        avg_angle = np.mean([e[4] for e in window])
        
        smoothed.append((avg_cx, avg_cy, avg_a, avg_b, avg_angle))
    
    return smoothed


def draw_eye_ellipse(
    frame: np.ndarray,
    cx: float,
    cy: float,
    a: float,
    b: float,
    angle: float,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw ellipse around eye."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    cx = int(cx)
    cy = int(cy)
    a = int(a)
    b = int(b)
    r = int(max(a, b))
    
    if 0 <= cx < w and 0 <= cy < h:
        box = ((cx, cy), (a * 2, b * 2), angle)
        cv2.ellipse(output, box, color, thickness)
        
        cv2.circle(output, (cx, cy), 3, color, -1)
        
        r = max(a, b) + 5
        cv2.line(output, (cx - r, cy), (cx - r + 8, cy), color, 1)
        cv2.line(output, (cx + r - 8, cy), (cx + r, cy), color, 1)
        cv2.line(output, (cx, cy - r), (cx, cy - r + 8), color, 1)
        cv2.line(output, (cx, cy + r - 8), (cx, cy + r), color, 1)
    
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Track eye with ellipse overlay (Step 2).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=5,
        help="Smoothing radius for ellipse parameters.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=12,
        help="Number of tracking points around ellipse.",
    )
    parser.add_argument(
        "--min-axis",
        type=int,
        default=8,
        help="Minimum semi-axis length.",
    )
    parser.add_argument(
        "--max-axis",
        type=int,
        default=60,
        help="Maximum semi-axis length.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Failed to open: {args.input}", file=sys.stderr)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_center = (w // 2, h // 2)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.output}", file=sys.stderr)
        cap.release()
        return 1

    max_frames = args.max_frames if args.max_frames else total_frames
    
    print(f"Processing {max_frames} frames at {w}x{h}...")
    
    ellipse_data = []
    prev_pts = None
    prev_gray = None
    first_frame = True
    
    for frame_idx in range(max_frames):
        _print_progress(frame_idx + 1, max_frames, "  ")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        gray, inverted = preprocess_for_eye_detection(frame)
        
        if first_frame:
            detected = detect_iris_ellipse(inverted, frame_center, 
                                          min_axis=args.min_axis, max_axis=args.max_axis)
            
            if detected is None:
                fallback = detect_iris_circle(inverted, frame_center)
                if fallback:
                    cx, cy, radius = fallback
                    a, b, angle = radius, radius, 0.0
                    print(f"Using circle fallback: center=({cx}, {cy}), radius={radius}")
                else:
                    cx, cy, a, b, angle = frame_center[0], frame_center[1], 25, 25, 0.0
                    print("Using frame center as fallback")
            else:
                cx, cy, a, b, angle = detected
            
            prev_pts = create_tracking_points(cx, cy, a, b, angle, args.num_points)
            prev_gray = gray.copy()
            first_frame = False
            
            ellipse_data.append((cx, cy, a, b, angle))
            frame_with_ellipse = draw_eye_ellipse(frame, cx, cy, a, b, angle)
            writer.write(frame_with_ellipse)
            continue
        
        tracked_prev, tracked_curr = track_with_optical_flow(prev_gray, gray, prev_pts)
        
        if tracked_curr is not None and len(tracked_curr) >= 5:
            cx, cy, a, b, angle = fit_ellipse_to_points(tracked_curr.reshape(-1, 2))
            prev_pts = create_tracking_points(cx, cy, a, b, angle, args.num_points)
        else:
            detected = detect_iris_ellipse(inverted, frame_center,
                                           min_axis=args.min_axis, max_axis=args.max_axis)
            if detected is not None:
                cx, cy, a, b, angle = detected
            else:
                fallback = detect_iris_circle(inverted, frame_center)
                if fallback:
                    cx, cy, radius = fallback
                    a, b, angle = radius, radius, 0.0
                elif ellipse_data:
                    cx, cy, a, b, angle = ellipse_data[-1]
                else:
                    cx, cy, a, b, angle = frame_center[0], frame_center[1], 25, 25, 0.0
            
            prev_pts = create_tracking_points(cx, cy, a, b, angle, args.num_points)
        
        ellipse_data.append((cx, cy, a, b, angle))
        prev_gray = gray.copy()
        
        frame_with_ellipse = draw_eye_ellipse(frame, cx, cy, a, b, angle)
        writer.write(frame_with_ellipse)

    cap.release()
    writer.release()
    
    if len(ellipse_data) > args.smooth_radius * 2:
        print(f"\nSmoothing {len(ellipse_data)} ellipse parameters...")
        smoothed = smooth_ellipses(ellipse_data, args.smooth_radius)
        
        print("Rewriting with smoothed ellipses...")
        
        cap = cv2.VideoCapture(str(args.input))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
        
        for frame_idx in range(min(max_frames, len(smoothed))):
            ok, frame = cap.read()
            if not ok:
                break
            
            cx, cy, a, b, angle = smoothed[frame_idx]
            frame_with_ellipse = draw_eye_ellipse(frame, cx, cy, a, b, angle)
            writer.write(frame_with_ellipse)
        
        cap.release()
        writer.release()

    print(f"\nProcessed {len(ellipse_data)} frames.")

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
