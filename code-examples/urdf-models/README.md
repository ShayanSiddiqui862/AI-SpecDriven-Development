# URDF Models for Humanoid Robots

This directory contains URDF models for different humanoid robot configurations used in the Physical AI & Humanoid Robotics textbook.

## Overview

This package provides URDF models for:
- Basic humanoid robot with 2 legs, 2 arms
- Joint configurations for walking and manipulation
- Sensor placements (IMU, cameras, LiDAR)
- Collision and visual meshes

## Models

- `simple_humanoid.urdf` - Basic humanoid model with essential joints
- `advanced_humanoid.urdf` - Full humanoid with all DOFs and sensors
- `humanoid.gazebo` - Gazebo-specific configurations

## Usage

```bash
# View the model in RViz
ros2 run rviz2 rviz2

# Launch with joint state publisher
ros2 launch urdf_models view_model.launch.py

# Simulate in Gazebo
ros2 launch urdf_models spawn_model.launch.py
```

## Joint Configuration

The humanoid robot includes:
- 6 DOF per leg (hip, knee, ankle)
- 7 DOF per arm (shoulder, elbow, wrist)
- 3 DOF for torso and head