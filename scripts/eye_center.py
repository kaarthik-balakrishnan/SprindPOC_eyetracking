#!/usr/bin/env python3
"""
Step 2: Eye Tracking with Circle Overlay

- Converts to grayscale and inverts to make pupil bright
- Uses center-of-image heuristic to locate eye
- Tracks using multi-point optical flow
- Draws circle on original video

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
    Returns both grayscale and inverted image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    inverted = cv2.bitwise_not(enhanced)
    
    return gray, inverted


def detect_eye_center(inverted: np.ndarray, frame_center: tuple[int, int]) -> tuple[int, int, int]:
    """
    Detect eye center using:
    1. Threshold to find bright regions (eye appears bright after inversion)
    2. Find largest bright blob
    3. If no good blob, fall back to center of image
    Returns (x, y, radius).
    """
    h, w = inverted.shape
    
    _, thresh = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = np.argmax(areas)
        contour = contours[max_idx]
        
        if cv2.contourArea(contour) > 100:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                center_dist = np.sqrt((cx - frame_center[0])**2 + (cy - frame_center[1])**2)
                max_dist = np.sqrt(frame_center[0]**2 + frame_center[1]**2)
                
                if center_dist < max_dist * 0.4:
                    return cx, cy, int(radius)
    
    return frame_center[0], frame_center[1], 30


def create_tracking_points(cx: int, cy: int, radius: int, num_points: int = 8) -> np.ndarray:
    """Create tracking points around the pupil perimeter."""
    points = []
    angle_step = 2 * np.pi / num_points
    
    for i in range(num_points):
        angle = i * angle_step
        
        px = int(cx + radius * np.cos(angle))
        py = int(cy + radius * np.sin(angle))
        points.append([px, py])
        
        px2 = int(cx + (radius * 0.6) * np.cos(angle))
        py2 = int(cy + (radius * 0.6) * np.sin(angle))
        points.append([px2, py2])
    
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
    
    if len(valid_next) < 4:
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


def compute_center_from_points(tracked_pts: np.ndarray) -> tuple[int, int, int]:
    """Compute pupil center and radius from tracked points."""
    if tracked_pts is None or len(tracked_pts) < 3:
        return 0, 0, 25
    
    pts = tracked_pts.reshape(-1, 2)
    
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    
    distances = np.sqrt((pts[:, 0] - center_x)**2 + (pts[:, 1] - center_y)**2)
    
    inlier_mask = distances < np.percentile(distances, 75)
    inlier_pts = pts[inlier_mask]
    
    if len(inlier_pts) >= 3:
        center_x = np.mean(inlier_pts[:, 0])
        center_y = np.mean(inlier_pts[:, 1])
        radius = np.median(np.sqrt((inlier_pts[:, 0] - center_x)**2 + (inlier_pts[:, 1] - center_y)**2))
    else:
        radius = np.median(distances)
    
    return int(center_x), int(center_y), max(int(radius), 15)


def smooth_positions(positions: list, radius: int = 5) -> list:
    """Smooth positions using moving average."""
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
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw circle around pupil."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    cx = int(cx)
    cy = int(cy)
    radius = int(radius)
    
    if 0 <= cx < w and 0 <= cy < h:
        cv2.circle(output, (cx, cy), radius, color, thickness)
        
        cv2.circle(output, (cx, cy), 3, color, -1)
        
        cv2.line(output, (cx - radius - 10, cy), (cx - radius - 3, cy), color, 1)
        cv2.line(output, (cx + radius + 3, cy), (cx + radius + 10, cy), color, 1)
        cv2.line(output, (cx, cy - radius - 10), (cx, cy - radius - 3), color, 1)
        cv2.line(output, (cx, cy + radius + 3), (cx, cy + radius + 10), color, 1)
    
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Track pupil and draw circle overlay (Step 2).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=5,
        help="Smoothing radius for positions.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8,
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.output}", file=sys.stderr)
        cap.release()
        return 1

    max_frames = args.max_frames if args.max_frames else total_frames
    
    print(f"Processing {max_frames} frames at {w}x{h}...")
    
    pupil_positions = []
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
            cx, cy, radius = detect_eye_center(inverted, frame_center)
            prev_pts = create_tracking_points(cx, cy, radius, args.num_points)
            prev_gray = gray.copy()
            first_frame = False
            
            pupil_positions.append((cx, cy, radius))
            frame_with_circle = draw_pupil_circle(frame, cx, cy, args.circle_radius)
            writer.write(frame_with_circle)
            continue
        
        tracked_prev, tracked_curr = track_with_optical_flow(prev_gray, gray, prev_pts)
        
        if tracked_curr is not None and len(tracked_curr) >= 3:
            cx, cy, detected_radius = compute_center_from_points(tracked_curr.reshape(-1, 2))
            prev_pts = create_tracking_points(cx, cy, detected_radius, args.num_points)
        else:
            cx, cy, detected_radius = detect_eye_center(inverted, frame_center)
            prev_pts = create_tracking_points(cx, cy, detected_radius, args.num_points)
        
        pupil_positions.append((cx, cy, detected_radius))
        prev_gray = gray.copy()
        
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
