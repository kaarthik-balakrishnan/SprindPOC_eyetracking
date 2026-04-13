# Step 2: Eye Tracking with Circle Overlay

## Overview

This step tracks the eye position across frames and renders a stable white circle overlay centered on the eye. Unlike the previous cropping approach, this preserves the full frame context while ensuring the circle remains aligned with the eye despite camera shake or head movement.

## Algorithm

### 1. Reference Frame Creation

Identical to Step 1: The first N frames are averaged to create a robust reference:

$$\bar{I} = \frac{1}{N} \sum_{i=1}^{N} I_i$$

The eye center is detected in this reference frame and stored as `(ref_eye_x, ref_eye_y)`.

### 2. Eye Center Detection

The eye center is detected using a **two-stage approach**:

#### Stage 1: Hough Circle Transform

The **Hough Circle Transform** detects circular patterns (iris/pupil):

$$(x - x_0)^2 + (y - y_0)^2 = r^2$$

Parameters:
- **dp = 1.0**: Accumulator resolution relative to image resolution
- **minDist = 30**: Minimum distance between detected circles
- **param1 = 50**: Upper threshold for Canny edge detector
- **param2 = 30**: Lower threshold for circle center voting
- **minRadius/maxRadius**: Search range (10-80 pixels at 720p)

#### Stage 2: Contour-Based Fallback

If HoughCircles fails, morphological operations extract dark regions:

$$I_{binary} = \text{THRESH}_{OTSU}(\text{Clahe}(\text{Blur}(I)))$$

```python
thresh = CLOSE(thresh, kernel_5x5)  # Fill gaps
thresh = OPEN(thresh, kernel_5x5)   # Remove noise
```

The largest contour's centroid is returned:

$$\bar{x} = \frac{M_{10}}{M_{00}}, \quad \bar{y} = \frac{M_{01}}{M_{00}}$$

where $M_{ij}$ are image moments.

### 3. Feature-Based Frame Alignment

Features are detected in a **cropped region around the eye** to focus tracking on relevant motion:

```python
x1 = ref_eye_x - crop_size
x2 = ref_eye_x + crop_size
y1 = ref_eye_y - crop_size
y2 = ref_eye_y + crop_size
```

**Shi-Tomasi corner detection** identifies trackable features:

$$R = \min(\lambda_1, \lambda_2) > \lambda_{threshold}$$

### 4. Optical Flow Tracking

**Lucas-Kanade optical flow** tracks features from reference to each frame:

$$\nabla I \cdot \mathbf{V} = -I_t$$

Features are filtered using RANSAC-based affine estimation to reject outliers.

### 5. Transform Smoothing

Identical to Step 1: A **causal moving average filter** smooths the transforms:

$$\tilde{T}_i = \frac{1}{2R+1} \sum_{j=i-R}^{i+R} T_j$$

### 6. Circle Overlay Rendering

A white circle is drawn at the **reference eye position** (not the detected position):

```python
cv2.circle(frame, (ref_eye_x, ref_eye_y), radius, (255, 255, 255), 2)
```

Crosshairs are added for visualization:
```python
# Horizontal line extensions
cv2.line(frame, (x - r - 10, y), (x - r - 5, y), color, 1)
cv2.line(frame, (x + r + 5, y), (x + r + 10, y), color, 1)
# Vertical line extensions
cv2.line(frame, (x, y - r - 10), (x, y - r - 5), color, 1)
cv2.line(frame, (x, y + r + 5), (x, y + r + 10), color, 1)
```

This ensures the circle appears **stable** relative to the eye, even if the detected position drifts.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ref-frames` | 5 | Frames to average for reference |
| `--smooth-radius` | 15 | Transform smoothing radius |
| `--crop-size` | 200 | Search region around eye (pixels) |
| `--circle-radius` | 40 | Radius of circle to draw |
| `--scale` | 1.0 | Scale factor for processing |
| `--max-frames` | all | Process only first N frames |

## Why Circle at Reference Position?

Drawing the circle at the **reference position** (rather than detected position) provides:

1. **Stability**: The circle doesn't jitter with detection noise
2. **Consistency**: All frames share the same coordinate system
3. **Drift compensation**: Detection errors don't accumulate

The alignment transform ensures the reference position corresponds to the actual eye location in each frame.

## Scientific Considerations

### Limitations
1. **Fixed circle size**: Assumes iris diameter is constant
2. **Single eye tracking**: Only one circle per frame
3. **2D projection**: No depth estimation in this step
4. **Occlusion handling**: Circle persists through blinks

### Improvements (Future)
- Adaptive circle size based on detected iris radius
- Multiple circle tracking (both eyes)
- Blink detection to hide circle during eye closure
- 3D pose estimation for perspective-correct overlay

## References

1. Kim, J., & Kwon, J. (2014). Real-time eye tracking using Hough circle transform.
2. Wang, H., et al. (2017). Eye tracking via optical flow and geometric constraints.
3. Bradski, G. (2000). OpenCV Library for Computer Vision Applications.
4. Sonka, M., Hlavac, V., & Boyle, R. (2014). Image Processing, Analysis, and Machine Vision.
