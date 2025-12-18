# Visual Simultaneous Localization and Mapping (VSLAM)

This directory contains hardware-accelerated VSLAM implementation with RealSense data for humanoid robotics applications.

## Overview

This package provides:
- ORB-SLAM2/ORB-SLAM3 implementation examples
- RealSense camera integration
- GPU-accelerated feature detection and matching
- Loop closure and map optimization
- Visual-inertial odometry examples

## Components

- `realsense_vslam.py` - RealSense camera VSLAM implementation
- `feature_detector.py` - Feature detection and description
- `pose_estimator.py` - Pose estimation from visual features
- `map_optimizer.py` - Map optimization and loop closure

## Dependencies

- OpenCV
- ORB-SLAM3
- pyrealsense2
- Pangolin (for visualization)

## Usage

```bash
# Run RealSense VSLAM
python3 realsense_vslam.py --config config.yaml

# Run feature detection example
python3 feature_detector.py --input sample_image.png
```