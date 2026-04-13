#!/usr/bin/env python3
"""
Step 1: Stabilize handheld video using tracked feature points and smoothed
affine transforms (translation + rotation + uniform scale).

Two-pass: (1) compute inter-frame transforms, (2) reread and warp — keeps RAM low.

Usage:
  python scripts/stabilize_video.py --input data/PXL_20260410_024928909.mp4 \\
    --output outputs/stabilized.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _affine_to_delta(M: np.ndarray) -> np.ndarray:
    """2x3 partial affine -> [dx, dy, da] where da is rotation in radians."""
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    da = float(np.arctan2(M[1, 0], M[0, 0]))
    return np.array([dx, dy, da], dtype=np.float64)


def _delta_to_affine(d: np.ndarray) -> np.ndarray:
    """[dx, dy, da] -> 2x3 partial affine (unit scale)."""
    dx, dy, da = d
    c, s = np.cos(da), np.sin(da)
    return np.array([[c, s, dx], [-s, c, dy]], dtype=np.float64)


def _moving_average(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x.copy()
    k = 2 * radius + 1
    pad = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = pad[i : i + k].mean(axis=0)
    return out


def compute_deltas(
    cap: cv2.VideoCapture,
    *,
    max_frames: int | None = None,
    max_corners: int = 200,
    quality: float = 0.01,
    min_distance: int = 30,
    block_size: int = 3,
    lk_win: tuple[int, int] = (21, 21),
    lk_max_level: int = 3,
) -> tuple[int, int, list[np.ndarray]]:
    """Returns (width, height, deltas) where deltas[k] is frame k -> k+1."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise RuntimeError("Could not read first frame.")

    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=block_size,
    )
    if prev_pts is None or len(prev_pts) < 8:
        raise RuntimeError("Not enough features in first frame; try lowering quality/minDistance.")

    if max_frames is not None and max_frames <= 1:
        return w, h, []

    deltas: list[np.ndarray] = []
    target_deltas = None if max_frames is None else max_frames - 1

    while True:
        if target_deltas is not None and len(deltas) >= target_deltas:
            break
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_pts,
            None,
            winSize=lk_win,
            maxLevel=lk_max_level,
        )
        if next_pts is None:
            break
        status = status.ravel().astype(bool)
        p0 = prev_pts[status]
        p1 = next_pts[status]

        if len(p0) < 8:
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max_corners,
                qualityLevel=quality,
                minDistance=min_distance,
                blockSize=block_size,
            )
            if prev_pts is None:
                break
            deltas.append(np.zeros(3, dtype=np.float64))
            continue

        M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
        if M is None:
            d = np.zeros(3, dtype=np.float64)
        else:
            d = _affine_to_delta(M)
        deltas.append(d)

        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_distance,
            blockSize=block_size,
        )
        if prev_pts is None:
            if inliers is not None and inliers.any():
                prev_pts = p1[inliers.ravel().astype(bool)]
            else:
                prev_pts = p1

    return w, h, deltas


def write_stabilized(
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    diffs: np.ndarray,
    *,
    crop_border: int,
    max_frames: int | None = None,
) -> None:
    """diffs[i] = smooth_traj[i] - traj[i]; one row per frame."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w - 2 * crop_border
    out_h = h - 2 * crop_border

    fi = 0
    limit = len(diffs) if max_frames is None else min(len(diffs), max_frames)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi >= limit:
            break
        T = _delta_to_affine(diffs[fi])
        stab = cv2.warpAffine(
            frame,
            T,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if crop_border > 0:
            stab = stab[crop_border : h - crop_border, crop_border : w - crop_border]
        writer.write(stab)
        fi += 1

    print(f"Wrote {fi} frames.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stabilize video (Step 1).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input MP4 path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    parser.add_argument("--smooth-radius", type=int, default=15, help="Moving-average half-width in frames.")
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Pixels to crop on each side after warp (removes black edges).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames (for quick tests).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap1 = cv2.VideoCapture(str(args.input))
    if not cap1.isOpened():
        print(f"Failed to open: {args.input}", file=sys.stderr)
        return 1

    try:
        w, h, deltas = compute_deltas(cap1, max_frames=args.max_frames)
    finally:
        cap1.release()

    if not deltas:
        traj = np.zeros((1, 3), dtype=np.float64)
        smooth = _moving_average(traj, args.smooth_radius)
        diffs = smooth - traj
    else:
        traj = np.zeros((len(deltas) + 1, 3), dtype=np.float64)
        for i, d in enumerate(deltas):
            traj[i + 1] = traj[i] + d

        smooth = _moving_average(traj, args.smooth_radius)
        diffs = smooth - traj

    cap2 = cv2.VideoCapture(str(args.input))
    if not cap2.isOpened():
        print(f"Failed to reopen: {args.input}", file=sys.stderr)
        return 1

    fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    out_w, out_h = w, h
    if args.crop_border > 0:
        out_w -= 2 * args.crop_border
        out_h -= 2 * args.crop_border
    if out_w <= 0 or out_h <= 0:
        print("crop_border too large for frame size.", file=sys.stderr)
        cap2.release()
        return 1

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.output}", file=sys.stderr)
        cap2.release()
        return 1

    try:
        write_stabilized(
            cap2,
            writer,
            diffs,
            crop_border=args.crop_border,
            max_frames=args.max_frames,
        )
    finally:
        cap2.release()
        writer.release()

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
