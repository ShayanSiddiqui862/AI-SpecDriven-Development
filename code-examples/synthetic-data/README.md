# Synthetic Data Generation Examples

This directory contains synthetic data generation pipelines for perception training in humanoid robotics.

## Overview

This package provides:
- 3D scene generation for training data
- Physics-based simulation of robot interactions
- Multi-sensor synthetic data generation
- Domain randomization techniques
- Annotation generation for training datasets

## Components

- `scene_generator.py` - 3D scene generation with randomization
- `sensor_simulator.py` - Multi-sensor data synthesis
- `domain_randomizer.py` - Domain randomization for sim-to-real
- `annotation_generator.py` - Automatic annotation generation

## Dependencies

- Blender or similar 3D engine
- Physics simulation library
- OpenCV
- NumPy

## Usage

```bash
# Generate synthetic training data
python3 scene_generator.py --output_dir ./training_data --num_scenes 1000

# Generate sensor data
python3 sensor_simulator.py --scene scene_001.json --sensors rgb depth lidar

# Apply domain randomization
python3 domain_randomizer.py --input_dir ./training_data --output_dir ./augmented_data
```