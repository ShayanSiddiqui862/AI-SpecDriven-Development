---
sidebar_position: 11
---

# ROS 2 Humble Installation Guide for Ubuntu 22.04

## Learning Objectives
By the end of this module, students will be able to:
- Install ROS 2 Humble Hawksbill on Ubuntu 22.04
- Configure the ROS 2 environment
- Verify the installation with basic tests
- Set up a ROS 2 workspace

## Prerequisites

- Ubuntu 22.04 LTS
- Internet connection
- Administrative privileges (sudo access)
- At least 5GB of free disk space
- Recommended: 8GB+ RAM for simulation work

## Theory

### ROS 2 Architecture
ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

ROS 2 uses DDS (Data Distribution Service) as its communication layer, providing reliable, real-time data exchange between nodes.

### Key Components
- **Nodes**: Basic compute elements that perform processing
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous, goal-oriented communication patterns
- **Packages**: Organizational units containing code, data, and configuration

## Installation Steps

### Step 1: Set Locale
Ensure your locale is set to support UTF-8:
```bash
locale  # Check current locale
# If not UTF-8, set it:
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Step 2: Setup Sources
Add the ROS 2 apt repository:
```bash
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Step 3: Install ROS 2 Packages
```bash
sudo apt update
sudo apt install ros-humble-desktop
```

Optional: Install additional development tools:
```bash
sudo apt install ros-dev-tools
```

### Step 4: Environment Setup
Source the ROS 2 environment:
```bash
source /opt/ros/humble/setup.bash
```

To automatically source the environment when opening a new terminal:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Implementation

### Creating a Workspace
1. Create the workspace directory:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

2. Build the workspace:
```bash
colcon build
```

3. Source the workspace:
```bash
source install/setup.bash
```

### Basic Verification
Test the installation with a simple publisher/subscriber example:

Terminal 1:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

Terminal 2:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

You should see messages being published and received between the nodes.

### Package Management
- List available packages: `ros2 pkg list`
- Create a new package: `ros2 pkg create --build-type ament_python my_robot_package`
- Find package info: `ros2 pkg info <package_name>`

## Exercises

1. Install ROS 2 Humble on your Ubuntu 22.04 system following the above steps
2. Create a new workspace and build it
3. Run the talker/listener example to verify communication
4. Create a simple ROS 2 package using the `ros2 pkg create` command
5. Explore the available ROS 2 packages with `ros2 pkg list`

## References

1. ROS 2 Documentation: https://docs.ros.org/en/humble/
2. ROS 2 Installation Guide: https://docs.ros.org/en/humble/Installation.html
3. ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html

## Further Reading

- ROS 2 Concepts: https://docs.ros.org/en/humble/Concepts.html
- ROS 2 Design: https://design.ros2.org/
- DDS (Data Distribution Service) specification