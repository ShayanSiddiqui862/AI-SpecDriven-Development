# Code Examples for Physical AI & Humanoid Robotics Textbook

This repository contains all code examples referenced in the Physical AI & Humanoid Robotics textbook. Each directory corresponds to examples from specific modules of the course.

## Repository Structure

### ROS 2 Examples
- `ros2-agents/` - Python agents using rclpy with 5+ working examples
- `urdf-models/` - URDF models for different humanoid robot configurations
- `bipedal-controllers/` - ROS 2 controllers for bipedal motion with parameter tuning

### Simulation Examples
- `gazebo-worlds/` - Gazebo world files for apartment environments and physics properties
- `sensor-sim/` - Sensor simulation examples with noise models (LiDAR, RGB-D, IMU)
- `unity-integration/` - Unity integration pipeline for human-robot interaction

### AI & Perception Examples
- `synthetic-data/` - Synthetic data generation pipelines for perception training
- `vslam/` - Hardware-accelerated VSLAM implementation with RealSense data
- `nav2-bipedal/` - Nav2 configuration for bipedal path planning with dynamic stability

### VLA System Examples
- `whisper-integration/` - Whisper integration for voice command processing in ROS 2
- `llm-prompts/` - LLM prompt engineering for task decomposition
- `multi-modal/` - Multi-modal fusion for vision grounding with language instructions

### Capstone Project
- `capstone/` - Complete "Autonomous Humanoid" capstone project implementation

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/physical-ai-robotics-examples.git
   ```

2. Navigate to the specific example directory you want to run

3. Follow the instructions in each directory's README file

## Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Gazebo Garden or higher
- Python 3.8 or higher
- Appropriate hardware for running examples (RTX GPU for Isaac Sim, sensors for perception examples)

## Contributing

We welcome contributions to improve the examples. Please follow the contribution guidelines in the main textbook repository.

## License

This code is provided as part of the Physical AI & Humanoid Robotics textbook under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.