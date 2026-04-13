# Step 1: Video Stabilization

## Overview

This step stabilizes handheld video footage by aligning all frames to a single reference frame, eliminating camera shake and jitter while preserving intentional camera motion.

## Algorithm

### 1. Reference Frame Creation

A robust reference frame is created by averaging the first N frames:

$$\bar{I} = \frac{1}{N} \sum_{i=1}^{N} I_i$$

where $N$ (default: 5) is chosen to balance noise reduction against temporal responsiveness. Averaging multiple frames:
- Reduces random sensor noise
- Produces a temporally stable reference
- Handles minor illumination variations

### 2. Feature Detection

Features are detected in the reference frame using **Shi-Tomasi corner detection**:

$$R = \min(\lambda_1, \lambda_2) > \lambda_{threshold}$$

where $\lambda_1, \lambda_2$ are eigenvalues of the structure tensor. Features are selected based on:
- **Quality level** (0.01): Minimum eigenvalue threshold
- **Minimum distance** (30px): Minimum separation between features
- **Max corners** (200): Maximum features to track

### 3. Optical Flow Tracking

Features are tracked to each subsequent frame using **Lucas-Kanade optical flow**:

$$\nabla I \cdot \mathbf{V} = -I_t$$

where $\mathbf{V} = (u, v)$ is the optical flow velocity vector.

**Parameters:**
- **Window size**: 21×21 pixels
- **Pyramid levels**: 3 (for handling large motions)

### 4. Transform Estimation

Feature correspondences are used to estimate a **similarity transform** (allowing rotation, translation, and uniform scale):

$$\mathbf{x}' = s \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \mathbf{x} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$

This is implemented via `cv2.estimateAffinePartial2D()` with **RANSAC** for robustness against outliers (incorrect correspondences).

### 5. Transform Smoothing

Individual transforms are smoothed using a **moving average filter**:

$$\tilde{T}_i = \frac{1}{2R+1} \sum_{j=i-R}^{i+R} T_j$$

where $R$ is the smoothing radius (default: 15 frames). This:
- Reduces high-frequency jitter
- Preserves intentional camera movements
- Introduces slight temporal lag

### 6. Frame Alignment

Each frame is warped to align with the reference using the inverse transform:

$$I_{stabilized} = I \cdot \tilde{T}^{-1}$$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ref-frames` | 5 | Number of frames to average for reference |
| `--smooth-radius` | 15 | Smoothing window radius (frames) |
| `--crop-border` | 0 | Pixels to crop from edges |
| `--scale` | 1.0 | Scale factor for downscaling |
| `--max-frames` | all | Process only first N frames |

## Output

- **Format**: MP4 video with H.264 compression
- **Resolution**: Same as input (or scaled)
- **Frame rate**: Same as input

## Scientific Considerations

### Limitations
1. **Drift**: Accumulated errors over long videos
2. **Occlusions**: Features lost when eye/face moves significantly
3. **Large motions**: Pyramid levels may not handle extreme shake
4. **Non-rigid motion**: Assumes planar scene (limited for close-up eye)

### Improvements (Future)
- Bundle adjustment across all frames
- Kalman filtering for motion prediction
- Deep learning-based stabilization (e.g., DUT, GAN-based)
- IMU integration if gyroscope data available

## References

1. Shi, J., & Tomasi, C. (1994). Good features to track.
2. Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique.
3. Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus paradigm.
