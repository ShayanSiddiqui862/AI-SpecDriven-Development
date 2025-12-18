# Bipedal Controllers

This directory contains ROS 2 controllers for bipedal motion with parameter tuning examples for humanoid robots.

## Overview

This package provides controllers for bipedal locomotion including:
- Joint position controllers
- PID controllers for balance
- Walking pattern generators
- Balance feedback controllers

## Prerequisites

- ROS 2 Humble Hawksbill
- ros2_control packages
- controller_manager
- joint_state_broadcaster

## Usage

```bash
# Build the package
colcon build --packages-select bipedal_controllers

# Source the workspace
source install/setup.bash

# Launch the controller
ros2 launch bipedal_controllers bipedal_controller.launch.py
```

## Configuration

Controllers can be configured through YAML parameter files that define:
- Joint limits
- PID gains
- Balance parameters
- Walking gait parameters