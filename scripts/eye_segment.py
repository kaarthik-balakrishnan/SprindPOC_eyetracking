#!/usr/bin/env python3
"""
Step 2: Eye Tracking with Color Segmentation

- Segments image by color regions
- Identifies eye region based on color properties
- Works on stabilized video without tracking from previous frames

Usage:
  python scripts/eye_segment.py --input outputs/stabilized.mp4 \\
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


def select_eye_region(frame: np.ndarray, window_name: str = "Select Eye Region") -> tuple[int, int, int, int] | None:
    """Let user draw a rectangle around the eye region. Returns (x, y, w, h) or None."""
    clone = frame.copy()
    
    rect_start = [None, None]
    rect_end = [None, None]
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, rect_start, rect_end
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_start[0] = x
            rect_start[1] = y
            rect_end[0] = x
            rect_end[1] = y
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                rect_end[0] = x
                rect_end[1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            rect_end[0] = x
            rect_end[1] = y
            drawing = False
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = clone.copy()
        if rect_start[0] is not None:
            cv2.rectangle(display, 
                         (min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])),
                         (max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])),
                         (0, 255, 0), 2)
        
        cv2.putText(display, "Draw rectangle around eye, press ENTER when done", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        elif key == 13:  # ENTER
            if rect_start[0] is not None:
                cv2.destroyWindow(window_name)
                x1 = min(rect_start[0], rect_end[0])
                y1 = min(rect_start[1], rect_end[1])
                x2 = max(rect_start[0], rect_end[0])
                y2 = max(rect_start[1], rect_end[1])
                return x1, y1, x2 - x1, y2 - y1
    
    cv2.destroyWindow(window_name)
    return None


def get_eye_color_profile(roi: np.ndarray) -> dict:
    """Extract color profile from eye ROI."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    h_mean = np.mean(hsv[:, :, 0])
    h_std = np.std(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    s_std = np.std(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    v_std = np.std(hsv[:, :, 2])
    
    return {
        'h_mean': h_mean, 'h_std': max(h_std, 20),
        's_mean': s_mean, 's_std': max(s_std, 30),
        'v_mean': v_mean, 'v_std': max(v_std, 30),
        'hsv': hsv
    }


def find_eye_by_color(
    frame: np.ndarray,
    profile: dict,
    search_region: tuple[int, int, int, int] | None = None,
    tolerance_scale: float = 1.5
) -> tuple[int, int] | None:
    """
    Find eye center by color matching.
    Returns (cx, cy) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if search_region:
        x, y, w, h = search_region
        hsv_search = hsv[y:y+h, x:x+w]
        frame_search = frame[y:y+h, x:x+w]
    else:
        hsv_search = hsv
        frame_search = frame
    
    h_tol = profile['h_std'] * tolerance_scale
    s_tol = profile['s_std'] * tolerance_scale
    v_tol = profile['v_std'] * tolerance_scale
    
    h_min = max(0, profile['h_mean'] - h_tol)
    h_max = min(180, profile['h_mean'] + h_tol)
    s_min = max(0, profile['s_mean'] - s_tol)
    s_max = min(255, profile['s_mean'] + s_tol)
    v_min = max(0, profile['v_mean'] - v_tol)
    v_max = min(255, profile['v_mean'] + v_tol)
    
    mask = cv2.inRange(hsv_search, (h_min, s_min, v_min), (h_max, s_max, v_max))
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_contour = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            if search_region:
                cx += x
                cy += y
            
            return cx, cy
    
    return None


def find_dark_region(
    frame: np.ndarray,
    search_region: tuple[int, int, int, int] | None = None,
    dark_threshold: int = 80
) -> tuple[int, int] | None:
    """
    Find pupil/iris as dark region within search area.
    Returns (cx, cy) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if search_region:
        x, y, w, h = search_region
        gray_search = gray[y:y+h, x:x+w]
        frame_search = frame[y:y+h, x:x+w]
    else:
        gray_search = gray
        frame_search = frame
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_search)
    
    _, dark_mask = cv2.threshold(enhanced, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    dark_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dark_regions.append((cx, cy, area))
    
    if not dark_regions:
        return None
    
    dark_regions.sort(key=lambda r: r[2], reverse=True)
    
    best = dark_regions[0]
    if search_region:
        return best[0] + x, best[1] + y
    return best[0], best[1]


def find_eye_by_combined(
    frame: np.ndarray,
    color_profile: dict | None,
    search_region: tuple[int, int, int, int] | None,
    radius: int = 80
) -> tuple[int, int] | None:
    """
    Combine color and dark region detection.
    """
    dark_pos = find_dark_region(frame, search_region)
    
    if dark_pos:
        return dark_pos
    
    if color_profile:
        color_pos = find_eye_by_color(frame, color_profile, search_region)
        if color_pos:
            return color_pos
    
    if search_region:
        x, y, w, h = search_region
        return x + w // 2, y + h // 2
    
    h, w = frame.shape[:2]
    return w // 2, h // 2


def segment_image_kmeans(
    frame: np.ndarray,
    k: int = 5,
    attempts: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment image using K-means clustering.
    Returns (labels, centers).
    """
    h, w = frame.shape[:2]
    pixels = frame.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    
    labels = labels.reshape(h, w)
    
    return labels, centers.astype(np.uint8)


def find_eye_cluster(
    frame: np.ndarray,
    roi: np.ndarray,
    k: int = 5
) -> int:
    """
    Find which cluster the eye ROI belongs to.
    """
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_mean_hsv = np.mean(roi_hsv, axis=(0, 1))
    
    labels, centers = segment_image_kmeans(frame, k=k)
    
    best_cluster = 0
    best_dist = float('inf')
    
    for i, center_bgr in enumerate(centers):
        center_hsv = cv2.cvtColor(np.uint8([[center_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        dist = np.sum((center_hsv.astype(float) - roi_mean_hsv) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_cluster = i
    
    return best_cluster


def track_with_kmeans(
    frame: np.ndarray,
    eye_cluster: int,
    prev_center: tuple[int, int],
    radius: int = 100
) -> tuple[int, int] | None:
    """
    Track eye using K-means segmentation from previous position.
    """
    labels, centers = segment_image_kmeans(frame, k=5)
    
    x, y = prev_center
    
    search_x1 = max(0, x - radius)
    search_x2 = min(frame.shape[1], x + radius)
    search_y1 = max(0, y - radius)
    search_y2 = min(frame.shape[0], y + radius)
    
    search_labels = labels[search_y1:search_y2, search_x1:search_x2]
    
    mask = (search_labels == eye_cluster).astype(np.uint8) * 255
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_contour = None
    best_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best_contour = cnt
    
    if best_contour:
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) + search_x1
            cy = int(M["m01"] / M["m00"]) + search_y1
            return cx, cy
    
    return None


def draw_eye_tracking(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int = 30,
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """Draw eye tracking overlay."""
    output = frame.copy()
    
    cv2.circle(output, (cx, cy), radius, color, 2)
    cv2.circle(output, (cx, cy), 3, color, -1)
    
    r = radius + 10
    cv2.line(output, (cx - r, cy), (cx - r + 15, cy), color, 2)
    cv2.line(output, (cx + r - 15, cy), (cx + r, cy), color, 2)
    cv2.line(output, (cx, cy - r), (cx, cy - r + 15), color, 2)
    cv2.line(output, (cx, cy + r - 15), (cx, cy + r), color, 2)
    
    return output


def auto_detect_eye(frame: np.ndarray) -> tuple[int, int]:
    """
    Auto-detect eye by finding dark regions in center of frame.
    Returns (cx, cy).
    """
    h, w = frame.shape[:2]
    
    center_x, center_y = w // 2, h // 2
    
    search_w, search_h = w // 3, h // 3
    search_x = center_x - search_w // 2
    search_y = center_y - search_h // 2
    
    search_region = (search_x, search_y, search_w, search_h)
    
    dark_pos = find_dark_region(frame, search_region)
    
    if dark_pos:
        return dark_pos
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dark_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < (w * h * 0.1):
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                if dist < w * 0.3:
                    dark_regions.append((cx, cy, area, dist))
    
    if dark_regions:
        dark_regions.sort(key=lambda r: (r[3], -r[2]))
        return dark_regions[0][0], dark_regions[0][1]
    
    return center_x, center_y


def main() -> int:
    parser = argparse.ArgumentParser(description="Track eye with color segmentation (Step 2).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument("--radius", type=int, default=40, help="Eye radius for visualization.")
    parser.add_argument("--search-radius", type=int, default=150, help="Search radius around last position.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames.")
    parser.add_argument("--kmeans-k", type=int, default=5, help="Number of K-means clusters.")
    parser.add_argument("--save-center", type=Path, default=None, help="Save center to file.")
    parser.add_argument("--load-center", type=Path, default=None, help="Load center from file.")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect eye on first frame.")
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
    
    print(f"Reading first frame...")
    ok, first_frame = cap.read()
    if not ok:
        print("Failed to read first frame", file=sys.stderr)
        return 1
    
    eye_cluster = None
    
    if args.load_center:
        with open(args.load_center) as f:
            parts = f.read().strip().split(',')
            if len(parts) >= 2:
                cx, cy = int(parts[0]), int(parts[1])
            else:
                cx, cy = w // 2, h // 2
        roi_x, roi_y = cx - 50, cy - 50
        roi_w, roi_h = 100, 100
        print(f"Loaded center: ({cx}, {cy})")
    elif args.auto_detect:
        print("Auto-detecting eye on first frame...")
        cx, cy = auto_detect_eye(first_frame)
        roi_x, roi_y = cx - 50, cy - 50
        roi_w, roi_h = 100, 100
        print(f"Auto-detected center: ({cx}, {cy})")
    else:
        print("Select eye region by drawing a rectangle...")
        region = select_eye_region(first_frame)
        if region is None:
            print("No region selected, exiting.")
            return 1
        
        roi_x, roi_y, roi_w, roi_h = region
        cx = roi_x + roi_w // 2
        cy = roi_y + roi_h // 2
        print(f"Selected region: ({roi_x}, {roi_y}, {roi_w}, {roi_h}), center: ({cx}, {cy})")
        
        eye_roi = first_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        eye_cluster = find_eye_cluster(first_frame, eye_roi, k=args.kmeans_k)
        print(f"Eye cluster: {eye_cluster}")
    
    if args.save_center:
        with open(args.save_center, 'w') as f:
            f.write(f"{cx},{cy}")
        print(f"Saved center to {args.save_center}")
    
    color_profile = get_eye_color_profile(first_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])
    print(f"Color profile - H:({color_profile['h_mean']:.1f}±{color_profile['h_std']:.1f}), "
          f"S:({color_profile['s_mean']:.1f}±{color_profile['s_std']:.1f}), "
          f"V:({color_profile['v_mean']:.1f}±{color_profile['v_std']:.1f})")
    
    center_data = [(cx, cy)]
    frame_with_overlay = draw_eye_tracking(first_frame, cx, cy, args.radius)
    writer.write(frame_with_overlay)
    
    prev_center = (cx, cy)
    
    print(f"Processing {max_frames - 1} frames...")
    
    for frame_idx in range(1, max_frames):
        _print_progress(frame_idx, max_frames - 1, "  ")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        search_region = (
            max(0, prev_center[0] - args.search_radius),
            max(0, prev_center[1] - args.search_radius),
            args.search_radius * 2,
            args.search_radius * 2
        )
        
        if eye_cluster is not None:
            tracked = track_with_kmeans(frame, eye_cluster, prev_center, radius=args.search_radius // 2)
        else:
            tracked = None
        
        if tracked is None:
            tracked = find_eye_by_combined(frame, color_profile, search_region)
        
        if tracked is None:
            tracked = find_eye_by_combined(frame, color_profile, None)
        
        if tracked is None:
            tracked = prev_center
        
        cx, cy = tracked
        prev_center = (cx, cy)
        center_data.append((cx, cy))
        
        frame_with_overlay = draw_eye_tracking(frame, cx, cy, args.radius)
        writer.write(frame_with_overlay)

    cap.release()
    writer.release()

    print(f"\nProcessed {len(center_data)} frames.")
    print(f"X range: {min(c[0] for c in center_data)} - {max(c[0] for c in center_data)}")
    print(f"Y range: {min(c[1] for c in center_data)} - {max(c[1] for c in center_data)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
