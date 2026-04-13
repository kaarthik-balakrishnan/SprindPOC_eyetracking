#!/usr/bin/env python3
"""
Step 4: 3D Sphere Fitting & Gaze Vector Estimation

- Fits 3D sphere to tracked pupil data
- Estimates gaze direction vector
- Outputs 3D coordinates and gaze vectors

Usage:
  python scripts/gaze_3d.py --input outputs/pupil_data.csv \\
    --output outputs/gaze_3d.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_pupil_data(csv_path: Path) -> list[dict]:
    """Load pupil tracking data from CSV."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'frame': int(row['frame']),
                'x': float(row['x']),
                'y': float(row['y']),
                'radius': float(row['radius']),
            })
    return data


def fit_sphere_least_squares(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Fit a sphere to 3D points using least squares.
    Sphere equation: (x - xc)^2 + (y - yc)^2 + (z - zc)^2 = r^2
    
    Returns: (center_x, center_y, center_z, radius)
    """
    A = np.column_stack([
        2 * x,
        2 * y,
        2 * z,
        np.ones(len(x))
    ])
    
    b = x**2 + y**2 + z**2
    
    result = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, zc, r_squared = result[0]
    
    radius = np.sqrt(xc**2 + yc**2 + zc**2 + r_squared)
    
    return xc, yc, zc, radius


def estimate_gaze_vector(
    pupil_x: float,
    pupil_y: float,
    pupil_radius: float,
    eye_center_x: float,
    eye_center_y: float,
    eye_center_z: float,
    sphere_radius: float,
    focal_length: float = 500.0,
) -> tuple:
    """
    Estimate gaze direction vector based on pupil position relative to eye center.
    
    Uses simplified eye model:
    - Eye sphere centered at (eye_center_x, eye_center_y, eye_center_z)
    - Pupil position relative to sphere center gives gaze direction
    
    Returns: (gx, gy, gz, theta, phi)
    """
    dx = pupil_x - eye_center_x
    dy = pupil_y - eye_center_y
    
    gx = dx / focal_length
    gy = dy / focal_length
    gz = np.sqrt(1 - gx**2 - gy**2) if gx**2 + gy**2 < 1 else 0
    
    norm = np.sqrt(gx**2 + gy**2 + gz**2)
    if norm > 0:
        gx, gy, gz = gx/norm, gy/norm, gz/norm
    
    theta = np.arctan2(gy, gx)
    phi = np.arccos(gz)
    
    return gx, gy, gz, np.degrees(theta), np.degrees(phi)


def smooth_data(data: list, window_size: int = 5) -> list:
    """Apply moving average smoothing to reduce noise."""
    if len(data) < window_size:
        return data
    
    smoothed = []
    half = window_size // 2
    
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        window = data[start:end]
        
        smoothed.append({
            'frame': data[i]['frame'],
            'x': np.mean([p['x'] for p in window]),
            'y': np.mean([p['y'] for p in window]),
            'radius': np.mean([p['radius'] for p in window]),
        })
    
    return smoothed


def main() -> int:
    parser = argparse.ArgumentParser(description="3D gaze estimation (Step 4).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CSV path from pupil tracking.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Smoothing window size (frames).",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=12.0,
        help="Estimated eye sphere radius (mm, for visualization only).",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=500.0,
        help="Camera focal length (pixels).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    pupil_data = load_pupil_data(args.input)
    
    valid_data = [p for p in pupil_data if p['x'] >= 0]
    
    if len(valid_data) < 10:
        print(f"Not enough valid pupil data: {len(valid_data)} frames", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(pupil_data)} frames, {len(valid_data)} with valid pupil data.")
    
    valid_data = smooth_data(valid_data, args.smooth_window)
    
    x = np.array([p['x'] for p in valid_data])
    y = np.array([p['y'] for p in valid_data])
    z = np.array([p['radius'] for p in valid_data]) * 10
    
    xc, yc, zc, sphere_r = fit_sphere_least_squares(x, y, z)
    
    print(f"\nFitted sphere center: ({xc:.1f}, {yc:.1f}, {zc:.1f})")
    print(f"Fitted sphere radius (pixel units): {sphere_r:.1f}")
    
    gaze_data = []
    
    for p in pupil_data:
        if p['x'] >= 0:
            gx, gy, gz, theta, phi = estimate_gaze_vector(
                p['x'], p['y'], p['radius'],
                xc, yc, zc,
                args.sphere_radius,
                args.focal_length,
            )
        else:
            gx, gy, gz, theta, phi = -1, -1, -1, -1, -1
        
        gaze_data.append({
            'frame': p['frame'],
            'pupil_x': p['x'],
            'pupil_y': p['y'],
            'pupil_radius': p['radius'],
            'gaze_x': gx,
            'gaze_y': gy,
            'gaze_z': gz,
            'theta_deg': theta,
            'phi_deg': phi,
        })
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'pupil_x', 'pupil_y', 'pupil_radius',
            'gaze_x', 'gaze_y', 'gaze_z', 'theta_deg', 'phi_deg'
        ])
        writer.writeheader()
        writer.writerows(gaze_data)
    
    valid_gaze = [g for g in gaze_data if g['gaze_x'] >= 0]
    
    print(f"\nOutput: {args.output}")
    print(f"Processed {len(gaze_data)} frames")
    print(f"Average gaze: ({np.mean([g['gaze_x'] for g in valid_gaze]):.3f}, "
          f"{np.mean([g['gaze_y'] for g in valid_gaze]):.3f}, "
          f"{np.mean([g['gaze_z'] for g in valid_gaze]):.3f})")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
