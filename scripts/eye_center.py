#!/usr/bin/env python3
"""
Step 2: Eye Tracking with Template Matching

- User clicks on first frame to select eye center
- Creates template from first frame region
- Tracks with optical flow
- Periodically verifies with template matching to prevent drift
- Re-syncs if optical flow and template match disagree

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


def select_eye_center(frame: np.ndarray, window_name: str = "Select Eye") -> tuple[int, int] | None:
    """Let user click to select eye center. Returns (x, y) or None."""
    clone = frame.copy()
    
    instruction = "Click on the eye center, then press ENTER or SPACE. Press 'q' to cancel."
    cv2.putText(clone, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    click_point = [None, None]
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point[0] = x
            click_point[1] = y
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = clone.copy()
        if click_point[0] is not None:
            cv2.circle(display, (click_point[0], click_point[1]), 10, (0, 255, 0), 2)
            cv2.circle(display, (click_point[0], click_point[1]), 2, (0, 255, 0), -1)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        elif key in (13, 32):  # ENTER or SPACE
            if click_point[0] is not None:
                cv2.destroyWindow(window_name)
                return click_point[0], click_point[1]
    
    cv2.destroyWindow(window_name)
    return None


def create_template(gray_frame: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    """Create a template image of the eye region from the first frame."""
    h, w = gray_frame.shape
    
    half = radius
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    
    template = gray_frame[y1:y2, x1:x2].copy()
    
    return template


def template_match(
    gray_frame: np.ndarray,
    template: np.ndarray,
    search_cx: int,
    search_cy: int,
    search_radius: int,
    threshold: float = 0.7
) -> tuple[int, int, float] | None:
    """
    Find best match of template within search region.
    Returns (best_cx, best_cy, similarity_score) or None if no good match.
    """
    h, w = gray_frame.shape
    t_h, t_w = template.shape
    
    x1 = max(0, search_cx - search_radius)
    x2 = min(w - t_w, search_cx + search_radius)
    y1 = max(0, search_cy - search_radius)
    y2 = min(h - t_h, search_cy + search_radius)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    search_region = gray_frame[y1:y2, x1:x2]
    
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val < threshold:
        return None
    
    best_cx = x1 + max_loc[0] + t_w // 2
    best_cy = y1 + max_loc[1] + t_h // 2
    
    return best_cx, best_cy, max_val


def create_tracking_points_circle(cx: int, cy: int, radius: int, num_points: int = 16) -> np.ndarray:
    """Create tracking points in a circle around center."""
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        px = int(cx + radius * np.cos(angle))
        py = int(cy + radius * np.sin(angle))
        points.append([px, py])
        
        if i % 2 == 0:
            inner_r = int(radius * 0.5)
            px2 = int(cx + inner_r * np.cos(angle))
            py2 = int(cy + inner_r * np.sin(angle))
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


def compute_center_from_points(pts: np.ndarray) -> tuple[float, float]:
    """Compute center from tracked points."""
    if pts is None or len(pts) == 0:
        return 0.0, 0.0
    pts = pts.reshape(-1, 2)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def draw_tracking_overlay(
    frame: np.ndarray,
    cx: float,
    cy: float,
    radius: int,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    template_matched: bool = False,
) -> np.ndarray:
    """Draw tracking circle and crosshairs."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    cx_i = int(cx)
    cy_i = int(cy)
    
    if 0 <= cx_i < w and 0 <= cy_i < h:
        color = (0, 255, 0) if not template_matched else (0, 255, 255)
        cv2.circle(output, (cx_i, cy_i), radius, color, thickness)
        cv2.circle(output, (cx_i, cy_i), 3, color, -1)
        
        r = radius + 8
        cv2.line(output, (cx_i - r, cy_i), (cx_i - r + 10, cy_i), color, 1)
        cv2.line(output, (cx_i + r - 10, cy_i), (cx_i + r, cy_i), color, 1)
        cv2.line(output, (cx_i, cy_i - r), (cx_i, cy_i - r + 10), color, 1)
        cv2.line(output, (cx_i, cy_i + r - 10), (cx_i, cy_i + r), color, 1)
        
        if template_matched:
            cv2.putText(output, "SYNC", (cx_i - 20, cy_i - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Track eye with template matching (Step 2).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument(
        "--radius",
        type=int,
        default=60,
        help="Tracking radius around selected center.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=16,
        help="Number of tracking points.",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=5,
        help="Smoothing radius for center positions.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames.",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=30,
        help="Frames between template matching verification.",
    )
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.7,
        help="Minimum template match score (0-1).",
    )
    parser.add_argument(
        "--sync-threshold",
        type=float,
        default=30.0,
        help="Max pixel distance before re-sync.",
    )
    parser.add_argument(
        "--save-center",
        type=Path,
        default=None,
        help="Save selected center to file for batch processing.",
    )
    parser.add_argument(
        "--load-center",
        type=Path,
        default=None,
        help="Load center from file instead of interactive selection.",
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
    
    if args.load_center:
        with open(args.load_center) as f:
            cx, cy = map(int, f.read().strip().split(','))
        print(f"Loaded center from {args.load_center}: ({cx}, {cy})")
        
        ok, first_frame = cap.read()
        if not ok:
            print("Failed to read first frame", file=sys.stderr)
            return 1
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        template = create_template(first_gray, cx, cy, args.radius)
    else:
        print(f"Reading first frame to select eye center...")
        ok, first_frame = cap.read()
        if not ok:
            print("Failed to read first frame", file=sys.stderr)
            return 1
        
        center = select_eye_center(first_frame)
        if center is None:
            print("No center selected, exiting.")
            return 1
        
        cx, cy = center
        print(f"Selected eye center: ({cx}, {cy})")
        
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        template = create_template(first_gray, cx, cy, args.radius)
        
        if args.save_center:
            with open(args.save_center, 'w') as f:
                f.write(f"{cx},{cy}")
            print(f"Saved center to {args.save_center}")

    print(f"Template size: {template.shape}, mean intensity: {template.mean():.1f}")
    
    prev_pts = create_tracking_points_circle(cx, cy, args.radius, args.num_points)
    print(f"Created {len(prev_pts)} tracking points at radius {args.radius}")
    
    prev_gray = first_gray.copy()
    
    center_data = [(cx, cy)]
    synced = [True]
    frame_with_overlay = draw_tracking_overlay(first_frame, cx, cy, args.radius, template_matched=True)
    writer.write(frame_with_overlay)
    
    print(f"Processing {max_frames - 1} remaining frames...")
    
    for frame_idx in range(1, max_frames):
        _print_progress(frame_idx, max_frames - 1, "  ")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        tracked_prev, tracked_curr = track_with_optical_flow(prev_gray, gray, prev_pts)
        
        template_matched = False
        
        if tracked_curr is not None and len(tracked_curr) >= 4:
            new_cx, new_cy = compute_center_from_points(tracked_curr)
            prev_pts = create_tracking_points_circle(int(new_cx), int(new_cy), args.radius, args.num_points)
        else:
            new_cx, new_cy = center_data[-1]
        
        if frame_idx % args.sync_interval == 0:
            match = template_match(
                gray, template, int(new_cx), int(new_cy),
                search_radius=args.radius * 2,
                threshold=args.template_threshold
            )
            
            if match:
                tm_cx, tm_cy, score = match
                dist = np.sqrt((new_cx - tm_cx)**2 + (new_cy - tm_cy)**2)
                
                if dist > args.sync_threshold:
                    print(f"\n  Re-syncing at frame {frame_idx}: OF=({new_cx:.0f},{new_cy:.0f}), TM=({tm_cx},{tm_cy}), dist={dist:.1f}, score={score:.2f}")
                    new_cx, new_cy = tm_cx, tm_cy
                    template_matched = True
                    prev_pts = create_tracking_points_circle(int(new_cx), int(new_cy), args.radius, args.num_points)
                    template = create_template(gray, int(new_cx), int(new_cy), args.radius)
                else:
                    template_matched = True
                    template = create_template(gray, int(new_cx), int(new_cy), args.radius)
        
        cx, cy = new_cx, new_cy
        center_data.append((cx, cy))
        synced.append(template_matched)
        prev_gray = gray.copy()
        
        frame_with_overlay = draw_tracking_overlay(frame, cx, cy, args.radius, template_matched=synced[-1])
        writer.write(frame_with_overlay)

    cap.release()
    writer.release()
    
    if len(center_data) > args.smooth_radius * 2:
        print(f"\nSmoothing {len(center_data)} positions...")
        smoothed = []
        half = args.smooth_radius
        for i in range(len(center_data)):
            start = max(0, i - half)
            end = min(len(center_data), i + half + 1)
            window = center_data[start:end]
            avg_cx = np.mean([p[0] for p in window])
            avg_cy = np.mean([p[1] for p in window])
            smoothed.append((avg_cx, avg_cy))
        
        print("Rewriting with smoothed positions...")
        
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
            
            scx, scy = smoothed[frame_idx]
            frame_with_overlay = draw_tracking_overlay(frame, scx, scy, args.radius, template_matched=synced[frame_idx])
            writer.write(frame_with_overlay)
        
        cap.release()
        writer.release()

    sync_count = sum(synced)
    print(f"\nProcessed {len(center_data)} frames, re-synced {sync_count} times.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
