# Sensor Simulation Examples

This directory contains sensor simulation examples with noise models for LiDAR, RGB-D, and IMU sensors used in humanoid robotics.

## Overview

This package provides:
- LiDAR sensor simulation with realistic noise models
- RGB-D camera simulation with depth noise
- IMU sensor simulation with drift and noise
- Sensor fusion examples

## Components

- `lidar_sim.py` - LiDAR sensor simulation with configurable noise
- `camera_sim.py` - RGB-D camera simulation with distortion models
- `imu_sim.py` - IMU sensor simulation with bias and drift
- `sensor_fusion.py` - Example sensor fusion implementation

## Usage

```bash
# Run LiDAR simulation
python3 lidar_sim.py

# Run camera simulation
python3 camera_sim.py

# Run sensor fusion example
python3 sensor_fusion.py
```