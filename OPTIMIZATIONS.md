# Eye Tracking Pipeline - Optimization Recommendations

## Priority Order

1. Downscale to 720p
2. Add progress bar
3. Crop to eye ROI
4. Skip frames during stabilization
5. Add checkpointing
6. GPU acceleration (if Colab Pro)

---

## Step 1: Video Stabilization

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| **Downscale to 720p** before processing | ~10x faster | Low | ⬜ |
| Reduce `max_corners` (200→100) | 2x faster optical flow | Low | ⬜ |
| Increase `lk_win` (21→31) | Fewer iterations, faster | Low | ⬜ |
| Skip every 2nd frame for transform estimation | 2x faster | Low | ⬜ |
| Use GPU-accelerated OpenCV | 2-5x faster | Medium | ⬜ |
| Use `FFmpeg` for stabilization instead | Potentially faster | Medium | ⬜ |
| Add progress bar | Better UX | Low | ⬜ |

---

## Step 2: Eye Centering

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| Crop to ROI around eye (reduce search area) | 5-10x faster | Medium | ⬜ |
| Use face/eye detection cascade (Haar) | Auto-detect region | Medium | ⬜ |
| Downscale to 720p | ~10x faster | Low | ⬜ |
| Process every 5th frame for motion estimation | 5x faster | Low | ⬜ |

---

## Step 3: Contrast Enhancement

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| CLAHE on grayscale only | 3x faster than color | Low | ⬜ |
| Process only cropped eye region | 5-10x faster | Medium | ⬜ |
| Batch process with NumPy vectorization | 2-3x faster | Medium | ⬜ |
| Use GPU with Numba/Cupy | 10x faster | High | ⬜ |

---

## Step 4: Pupil Tracking

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| Use ELPupil method (deep learning) | More accurate | Medium | ⬜ |
| Downscale frames | ~10x faster | Low | ⬜ |
| Template matching with cached templates | 3x faster | Medium | ⬜ |
| Track at lower resolution, upscale results | 5x faster | Low | ⬜ |
| Use CUDA HoughCircles if available | 2x faster | Medium | ⬜ |
| Parallelize frame processing | 4-8x faster | High | ⬜ |

---

## Step 5: 3D Sphere & Gaze Vector

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| Skip frames where pupil not detected | Depends on data | Low | ⬜ |
| Batch compute sphere fits | 3x faster | Medium | ⬜ |
| Use NumPy for matrix operations | 2x faster | Low | ⬜ |
| GPU acceleration for least squares | 5x faster | Medium | ⬜ |

---

## Infrastructure

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| Use Colab Pro (better GPU/RAM) | 2-5x overall | Low | ⬜ |
| Pre-process video once, cache intermediate results | Depends on pipeline | Medium | ⬜ |
| Process in chunks, save checkpoints | Prevents data loss | Medium | ⬜ |
| Use efficient codec (H.264 output) | Smaller files, faster I/O | Low | ⬜ |

---

## Notes

<!-- Add implementation notes, results, and observations here -->

