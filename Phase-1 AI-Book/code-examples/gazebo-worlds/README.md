# Gazebo Worlds for Humanoid Robotics

This directory contains Gazebo world files for apartment environments and physics properties for humanoid robot simulation.

## Overview

This package provides:
- Apartment environment models
- Physics configurations for realistic simulation
- Humanoid robot spawn locations
- Sensor testing environments

## World Files

- `apartment_world.sdf` - Main apartment environment with furniture and obstacles
- `physics_config.sdf` - Physics engine configuration with appropriate parameters
- `test_course.world` - Simple course for testing robot navigation

## Usage

```bash
# Launch Gazebo with apartment world
gazebo --verbose apartment_world.sdf

# Or launch with ROS 2
ros2 launch gazebo_ros gazebo.launch.py world:=apartment_world.sdf
```

## Physics Configuration

The physics configuration includes:
- Appropriate gravity settings
- Realistic friction coefficients
- Stable simulation parameters for humanoid robots