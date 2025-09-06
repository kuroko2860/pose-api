# Multi-Person 3D Pose Estimation for Pistol Shooting & Martial Arts Training

This notebook implements advanced multi-person 3D pose estimation using multiple AI models for robust tracking in pistol shooting and martial arts training scenarios.

## Features:

- **Multi-person tracking** (2-3 people simultaneously)
- **3D pose estimation** with depth information
- **Stable tracking** with temporal smoothing
- **Hidden body part detection** using advanced models
- **Real-time processing** capabilities
- **Multiple model support** (MediaPipe, OpenPose, etc.)

## Models Used:

1. **MediaPipe Pose** - Fast and accurate 2D/3D pose estimation
2. **OpenPose** - Robust multi-person detection
3. **PoseNet** - Lightweight alternative
4. **Custom temporal filters** - For stability and smoothness

# 🧪 Testing and Usage Guide

## How to Test the System

### 1. **Test with Webcam (Real-time)**

```python
# Run this cell to test with your webcam
video_processor.process_webcam(duration=30)  # 30 seconds
```

### 2. **Test with Sample Video**

```python
# Create and process a sample video
video_processor.create_sample_video("test_video.mp4", duration=5)
results = video_processor.process_video_file("test_video.mp4", "output_with_poses.mp4")
```

### 3. **Test with Your Own Video**

```python
# Upload your video file to Colab and process it
results = video_processor.process_video_file("your_video.mp4", "output.mp4", max_frames=100)
```

### 4. **Test with Single Image**

```python
# Test with a single image
import cv2
image = cv2.imread("your_image.jpg")
result = pose_estimator.process_frame(image)
annotated = visualizer.draw_pose_2d(image, result['poses'])
cv2.imshow("Result", annotated)
```

### 5. **3D Visualization**

```python
# Create 3D plots of detected poses
if result['poses']:
    fig = visualizer.create_3d_plot(result['poses'])
    fig.show()
```

## Features Available:

- ✅ **Multi-person tracking** (2-3 people)
- ✅ **3D pose estimation** with depth
- ✅ **Stable tracking** with temporal smoothing
- ✅ **Hidden body part detection**
- ✅ **Shooting stance analysis**
- ✅ **Martial arts stance analysis**
- ✅ **Real-time processing**
- ✅ **Video processing**
- ✅ **3D visualization**
