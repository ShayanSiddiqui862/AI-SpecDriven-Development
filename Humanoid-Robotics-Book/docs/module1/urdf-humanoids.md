---
sidebar_position: 13
---

# URDF Tutorials for Humanoid Robot Models

## Learning Objectives
By the end of this module, students will be able to:
- Create URDF files for humanoid robot models
- Define links, joints, and materials for robot structures
- Implement visual and collision properties for simulation
- Configure transmissions for joint control
- Validate URDF models using ROS tools

## Theory

### URDF Overview
URDF (Unified Robot Description Format) is an XML format for representing robot models in ROS. It describes the physical and visual properties of a robot, including:
- Links: Rigid parts of the robot
- Joints: Connections between links
- Materials: Visual appearance
- Transmissions: Actuator interface definitions
- Gazebo plugins: Simulation-specific properties

### Key Components
- **Links**: Rigid bodies with visual, collision, and inertial properties
- **Joints**: Define how links move relative to each other (revolute, prismatic, fixed, etc.)
- **Materials**: Define visual appearance (color, texture)
- **Transmissions**: Define how actuators connect to joints

## Implementation

### Prerequisites
- ROS 2 Humble installed
- Basic understanding of robot kinematics
- Knowledge of XML syntax

### Model 1: Simple Humanoid Robot

#### Basic URDF Structure
Create `simple_humanoid.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.1"/>
  </joint>

  <!-- Left leg -->
  <link name="left_leg">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_leg_joint" type="fixed">
    <parent link="base_link"/>
    <child link="left_leg"/>
    <origin xyz="-0.05 0.0 -0.2"/>
  </joint>

  <!-- Right leg -->
  <link name="right_leg">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_leg_joint" type="fixed">
    <parent link="base_link"/>
    <child link="right_leg"/>
    <origin xyz="0.05 0.0 -0.2"/>
  </joint>

  <!-- Left arm -->
  <link name="left_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.00005"/>
    </inertial>
  </link>

  <joint name="left_arm_joint" type="fixed">
    <parent link="base_link"/>
    <child link="left_arm"/>
    <origin xyz="-0.15 0.0 0.0"/>
  </joint>

  <!-- Right arm -->
  <link name="right_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.00005"/>
    </inertial>
  </link>

  <joint name="right_arm_joint" type="fixed">
    <parent link="base_link"/>
    <child link="right_arm"/>
    <origin xyz="0.15 0.0 0.0"/>
  </joint>
</robot>
```

#### Testing the Simple Model
1. Create a launch file to visualize the model:
```xml
<launch>
  <param name="robot_description" command="xacro $(find-pkg-share your_robot_description)/urdf/simple_humanoid.urdf" />
  <node pkg="robot_state_publisher" executable="robot_state_publisher" name="robot_state_publisher">
    <param name="publish_frequency" value="30"/>
  </node>
  <node pkg="rviz2" executable="rviz2" name="rviz2" output="screen"/>
</launch>
```

2. Validate the URDF:
```bash
check_urdf simple_humanoid.urdf
```

### Model 2: Advanced Humanoid Robot with Actuated Joints

#### Enhanced URDF with Joint Control
Create `advanced_humanoid.urdf`:

```xml
<?xml version="1.0"?>
<robot name="advanced_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
  <material name="green">
    <color rgba="0 1 0 1"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.15 0.1 0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.1 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.06"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0002" ixy="0.0" ixz="0.0" iyy="0.0002" iyz="0.0" izz="0.0002"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.15"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Left hip -->
  <link name="left_hip">
    <visual>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="-0.08 0.0 -0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Left upper leg -->
  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.006" ixy="0.0" ixz="0.0" iyy="0.006" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.0 0.0 -0.15"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Left lower leg -->
  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.6"/>
      <inertia ixx="0.0045" ixy="0.0" ixz="0.0" iyy="0.0045" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0.0 0.0 -0.15"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Left foot -->
  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.4"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0012" iyz="0.0" izz="0.0015"/>
    </inertial>
  </link>

  <joint name="left_foot_joint" type="fixed">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0.0 0.0 -0.1"/>
  </joint>

  <!-- Right leg (mirror of left) -->
  <link name="right_hip">
    <visual>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip"/>
    <origin xyz="0.08 0.0 -0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.006" ixy="0.0" ixz="0.0" iyy="0.006" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_hip"/>
    <child link="right_upper_leg"/>
    <origin xyz="0.0 0.0 -0.15"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.6"/>
      <inertia ixx="0.0045" ixy="0.0" ixz="0.0" iyy="0.0045" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0.0 0.0 -0.15"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.4"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0012" iyz="0.0" izz="0.0015"/>
    </inertial>
  </link>

  <joint name="right_foot_joint" type="fixed">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0.0 0.0 -0.1"/>
  </joint>

  <!-- Left arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder"/>
    <origin xyz="-0.1 0.0 0.05"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0026" ixy="0.0" ixz="0.0" iyy="0.0026" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.0 0.0 -0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.4"/>
      <inertia ixx="0.0017" ixy="0.0" ixz="0.0" iyy="0.0017" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0.0 0.0 -0.125"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Right arm (mirror of left) -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.06 0.06 0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_shoulder"/>
    <origin xyz="0.1 0.0 0.05"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0026" ixy="0.0" ixz="0.0" iyy="0.0026" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.0 0.0 -0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.4"/>
      <inertia ixx="0.0017" ixy="0.0" ixz="0.0" iyy="0.0017" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_wrist_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0.0 0.0 -0.125"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Transmissions for joint control -->
  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_knee_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_ankle_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_ankle_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_ankle_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_knee_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_knee_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_ankle_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_ankle_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_ankle_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_shoulder_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_elbow_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_elbow_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_wrist_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_wrist_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_wrist_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_shoulder_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_shoulder_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_elbow_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_elbow_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_wrist_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wrist_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_wrist_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="neck_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="neck_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="neck_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo plugin for physics simulation -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/advanced_humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Gazebo materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_leg">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="right_leg">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="left_arm">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="right_arm">
    <material>Gazebo/Green</material>
  </gazebo>
</robot>
```

#### Testing the Advanced Model
1. Create a launch file for the advanced model:
```xml
<launch>
  <param name="robot_description" command="xacro $(find-pkg-share your_robot_description)/urdf/advanced_humanoid.urdf" />
  <node pkg="robot_state_publisher" executable="robot_state_publisher" name="robot_state_publisher">
    <param name="publish_frequency" value="30"/>
  </node>
  <node pkg="rviz2" executable="rviz2" name="rviz2" output="screen"/>
</launch>
```

2. Test with joint state publisher:
```bash
ros2 run joint_state_publisher joint_state_publisher
```

### URDF Best Practices

#### 1. Proper Inertial Properties
- Calculate masses and inertias accurately
- Use consistent units (SI units)
- Ensure inertial tensors are physically valid

#### 2. Collision vs Visual Geometry
- Use simpler geometry for collision detection
- Use detailed geometry for visual representation
- Align collision and visual origins properly

#### 3. Joint Limits and Dynamics
- Set appropriate joint limits based on physical constraints
- Include damping and friction parameters
- Use realistic effort and velocity limits

#### 4. Naming Conventions
- Use consistent naming for links and joints
- Follow ROS conventions (snake_case)
- Use descriptive names that reflect function

## Exercises

1. Create both humanoid models and visualize them in RViz
2. Add sensors (IMU, cameras) to the advanced model
3. Create a walking gait pattern using the joint controllers
4. Implement a simple walking controller that moves the robot forward
5. Add a controller to the neck joint to make the head follow a target
6. Create a launch file that starts both the robot state publisher and joint state publisher

## References

1. URDF Documentation: http://wiki.ros.org/urdf
2. URDF/XML Format: http://wiki.ros.org/urdf/XML
3. Robot State Publisher: http://wiki.ros.org/robot_state_publisher
4. Gazebo Integration: http://gazebosim.org/tutorials?tut=ros2_overview

## Further Reading

- Xacro: XML Macros for URDF (to reduce repetition)
- Joint limits and safety controllers
- URDF validation tools and debugging techniques
- Integration with MoveIt! for motion planning