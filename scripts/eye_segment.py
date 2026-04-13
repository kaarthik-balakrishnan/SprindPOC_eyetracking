#!/usr/bin/env python3
"""
Step 2: Eye Tracking with Template Matching

- Auto-detects eye on first frame as dark region closest to center
- Creates template from detected eye region
- Tracks using template matching throughout video
- No user input required

Usage:
  python scripts/eye_segment.py --input outputs/stabilized.mp4 \\
    --output outputs/eye_tracked.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _print_progress(current, total, prefix="  ", bar_length=40):
    if total <= 0:
        return
    percent = current / total
    filled = int(bar_length * percent)
    bar = "=" * filled + "-" * (bar_length - filled)
    elapsed = time.time() - _print_progress.start_time
    fps = current / elapsed if elapsed > 0 else 0
    sys.stdout.write(
        f"\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%) "
        f"Elapsed: {elapsed:.1f}s, FPS: {fps:.1f}"
    )
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")

_print_progress.start_time = time.time()


def find_darkest_region_center(
    frame: np.ndarray,
    search_x: int,
    search_y: int,
    search_w: int,
    search_h: int
) -> tuple[int, int, int]:
    """
    Find the darkest region within search area.
    Returns (cx, cy, area).
    """
    h, w = frame.shape[:2]
    
    x1 = max(0, search_x)
    y1 = max(0, search_y)
    x2 = min(w, search_x + search_w)
    y2 = min(h, search_y + search_h)
    
    if x2 <= x1 or y2 <= y1:
        return w // 2, h // 2, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)
    
    _, dark_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_x, center_y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
    
    best = None
    best_score = float('inf')
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
            
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            score = dist + 1000 / (area + 1)
            
            if score < best_score:
                best_score = score
                best = (cx, cy, area)
    
    if best:
        return best
    
    return center_x, center_y, 0


def create_template(gray: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    """Create template from eye region."""
    h, w = gray.shape
    
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius)
    
    template = gray[y1:y2, x1:x2].copy()
    
    return template


def match_template(
    gray: np.ndarray,
    template: np.ndarray,
    search_cx: int,
    search_cy: int,
    search_radius: int,
    threshold: float = 0.5
) -> tuple[int, int, float] | None:
    """
    Find template in search area.
    Returns (cx, cy, score) or None.
    """
    h, w = gray.shape
    t_h, t_w = template.shape
    
    margin = max(t_h, t_w) // 2
    x1 = max(0, search_cx - search_radius - margin)
    x2 = min(w - t_w, search_cx + search_radius + margin)
    y1 = max(0, search_cy - search_radius - margin)
    y2 = min(h - t_h, search_cy + search_radius + margin)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    search_region = gray[y1:y2, x1:x2]
    
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val < threshold:
        return None
    
    best_cx = x1 + max_loc[0] + t_w // 2
    best_cy = y1 + max_loc[1] + t_h // 2
    
    return best_cx, best_cy, max_val


def draw_tracking(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int = 40,
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """Draw tracking overlay."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    cx = int(cx)
    cy = int(cy)
    
    if 0 <= cx < w and 0 <= cy < h:
        cv2.circle(output, (cx, cy), radius, color, 2)
        cv2.circle(output, (cx, cy), 3, color, -1)
        
        r = radius + 10
        cv2.line(output, (cx - r, cy), (cx - r + 15, cy), color, 2)
        cv2.line(output, (cx + r - 15, cy), (cx + r, cy), color, 2)
        cv2.line(output, (cx, cy - r), (cx, cy - r + 15), color, 2)
        cv2.line(output, (cx, cy + r - 15), (cx, cy + r), color, 2)
    
    return output


def main() -> int:
    start_time = time.time()
    _print_progress.start_time = start_time
    
    parser = argparse.ArgumentParser(description="Track eye with template matching (Step 2).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument("--radius", type=int, default=60, help="Eye radius for tracking.")
    parser.add_argument("--search-radius", type=int, default=100, help="Search radius around last position.")
    parser.add_argument("--template-threshold", type=float, default=0.5, help="Template match threshold.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames.")
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
        print("Failed to open writer", file=sys.stderr)
        cap.release()
        return 1

    max_frames = args.max_frames if args.max_frames else total_frames
    
    print("Reading first frame to detect eye...")
    ok, first_frame = cap.read()
    if not ok:
        print("Failed to read first frame", file=sys.stderr)
        return 1
    
    center_x, center_y = w // 2, h // 2
    
    search_w, search_h = w // 3, h // 3
    search_x = center_x - search_w // 2
    search_y = center_y - search_h // 2
    
    cx, cy, area = find_darkest_region_center(first_frame, search_x, search_y, search_w, search_h)
    print(f"Detected eye center: ({cx}, {cy}), area: {area}")
    
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    template = create_template(first_gray, cx, cy, args.radius)
    print(f"Template size: {template.shape}")
    
    center_data = [(cx, cy)]
    frame_with_overlay = draw_tracking(first_frame, cx, cy, args.radius)
    writer.write(frame_with_overlay)
    
    prev_cx, prev_cy = cx, cy
    
    print(f"Processing {max_frames - 1} frames...")
    
    for frame_idx in range(1, max_frames):
        _print_progress(frame_idx, max_frames - 1)
        
        ok, frame = cap.read()
        if not ok:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        match = match_template(
            gray, template, prev_cx, prev_cy,
            search_radius=args.search_radius,
            threshold=args.template_threshold
        )
        
        if match:
            cx, cy, score = match
        else:
            cx, cy, _ = find_darkest_region_center(frame, search_x, search_y, search_w, search_h)
            template = create_template(gray, cx, cy, args.radius)
        
        prev_cx, prev_cy = cx, cy
        center_data.append((cx, cy))
        
        frame_with_overlay = draw_tracking(frame, cx, cy, args.radius)
        writer.write(frame_with_overlay)

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Processed frames: {len(center_data)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average FPS: {len(center_data) / elapsed:.1f}")
    print(f"X range: {min(c[0] for c in center_data)} - {max(c[0] for c in center_data)}")
    print(f"Y range: {min(c[1] for c in center_data)} - {max(c[1] for c in center_data)}")
    print(f"X std dev: {np.std([c[0] for c in center_data]):.2f} pixels")
    print(f"Y std dev: {np.std([c[1] for c in center_data]):.2f} pixels")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
