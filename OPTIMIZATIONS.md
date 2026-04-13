# Eye Tracking Pipeline - Optimization Recommendations

## Priority Order

1. ~~Downscale to 720p~~ ✅ Done
2. ~~Add progress bar~~ ✅ Done
3. ~~Crop to eye ROI~~ ✅ Done
4. Skip frames during stabilization
5. Add checkpointing
6. GPU acceleration (if Colab Pro)

---

## Step 1: Video Stabilization

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| ~~**Downscale to 720p** before processing~~ | ~10x faster | Low | ✅ Done |
| ~~Add progress bar~~ | Better UX | Low | ✅ Done |
| Reduce `max_corners` (200→100) | 2x faster optical flow | Low | ⬜ |
| Increase `lk_win` (21→31) | Fewer iterations, faster | Low | ⬜ |
| Skip every 2nd frame for transform estimation | 2x faster | Low | ⬜ |
| Use GPU-accelerated OpenCV | 2-5x faster | Medium | ⬜ |
| Use `FFmpeg` for stabilization instead | Potentially faster | Medium | ⬜ |

---

## Step 2: Eye Centering

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| ~~Crop to ROI around eye~~ | 5-10x faster | Medium | ✅ Done |
| Use face/eye detection cascade (Haar) | Auto-detect region | Medium | ⬜ |
| ~~Inherits downscaled input~~ | ~10x faster | Low | ✅ Done |
| Process every 5th frame for motion estimation | 5x faster | Low | ⬜ |

---

## Step 3: Contrast Enhancement

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| ~~CLAHE on grayscale~~ | 3x faster than color | Low | ✅ Done |
| ~~Process only cropped eye region~~ | 5-10x faster | Medium | ✅ Done |
| Batch process with NumPy vectorization | 2-3x faster | Medium | ⬜ |
| Use GPU with Numba/Cupy | 10x faster | High | ⬜ |

---

## Step 4: Pupil Tracking

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| ~~HoughCircles implementation~~ | Standard approach | Medium | ✅ Done |
| ~~Track at downscaled resolution~~ | ~10x faster | Low | ✅ Done |
| Use ELPupil method (deep learning) | More accurate | Medium | ⬜ |
| Template matching with cached templates | 3x faster | Medium | ⬜ |
| Use CUDA HoughCircles if available | 2x faster | Medium | ⬜ |
| Parallelize frame processing | 4-8x faster | High | ⬜ |

---

## Step 5: 3D Sphere & Gaze Vector

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| ~~Skip frames where pupil not detected~~ | Depends on data | Low | ✅ Done |
| ~~Use NumPy for matrix operations~~ | 2x faster | Low | ✅ Done |
| Batch compute sphere fits | 3x faster | Medium | ⬜ |
| GPU acceleration for least squares | 5x faster | Medium | ⬜ |

---

## Infrastructure

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| Use Colab Pro (better GPU/RAM) | 2-5x overall | Low | ⬜ |
| ~~Cache intermediate results to Google Drive~~ | Depends on pipeline | Medium | ✅ Done |
| Process in chunks, save checkpoints | Prevents data loss | Medium | ⬜ |
| Use efficient codec (H.264 output) | Smaller files, faster I/O | Low | ⬜ |

---

## Implementation Summary

### Completed (9 items):
1. ✅ Downscale to 720p (--scale 0.25)
2. ✅ Progress bars for all scripts
3. ✅ Eye ROI cropping (--crop-size)
4. ✅ CLAHE contrast enhancement
5. ✅ HoughCircles pupil detection
6. ✅ Optical flow tracking
7. ✅ Skip invalid pupil frames
8. ✅ NumPy matrix operations
9. ✅ Save outputs to Google Drive

### Remaining (11 items):
1. ⬜ Skip frames during stabilization
2. ⬜ Reduce max_corners / increase lk_win
3. ⬜ Face/eye detection cascade
4. ⬜ GPU acceleration
5. ⬜ FFmpeg alternative
6. ⬜ Checkpointing
7. ⬜ H.264 codec
8. ⬜ ELPupil deep learning
9. ⬜ Template matching
10. ⬜ Batch sphere fitting
11. ⬜ Colab Pro

---

## Notes

<!-- Add implementation notes, results, and observations here -->

