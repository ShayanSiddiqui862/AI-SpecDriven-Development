---
sidebar_position: 23
---

# Sensor Simulation (LiDAR, RGB-D, IMU Noise Models)

## Learning Objectives
By the end of this module, students will be able to:
- Configure realistic noise models for LiDAR sensors in Gazebo
- Implement RGB-D camera simulation with appropriate noise characteristics
- Simulate IMU sensors with realistic bias and drift parameters
- Integrate multiple sensor types in a cohesive simulation environment
- Validate sensor simulation accuracy against real-world data

## Theory

### Sensor Simulation Fundamentals
Simulating sensors accurately is crucial for developing robust robotics algorithms. Real sensors have inherent limitations and noise characteristics that must be replicated in simulation to ensure effective transfer from simulation to reality.

### Key Concepts
- **Sensor Noise**: Random variations in sensor measurements
- **Bias**: Systematic errors that shift measurements consistently
- **Drift**: Slow changes in sensor characteristics over time
- **Resolution**: The smallest detectable change in measurement
- **Latency**: Delay between physical event and sensor output
- **Dynamic Range**: Range of measurable values

### Types of Sensor Noise
- **Gaussian Noise**: Random noise following a normal distribution
- **Quantization Noise**: Discretization errors due to finite resolution
- **Multiplicative Noise**: Noise that scales with signal strength
- **Impulsive Noise**: Occasional large errors (outliers)

## Implementation

### Prerequisites
- Gazebo Garden or higher installed
- ROS 2 Humble with Gazebo plugins
- Basic understanding of sensor physics and statistics

### LiDAR Sensor Simulation

#### 1. Creating a Realistic LiDAR Model
Create `models/lidar_3d/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="lidar_3d">
    <link name="lidar_link">
      <pose>0 0 0 0 0 0</pose>

      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.001</iyy>
          <iyz>0.0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>

      <collision name="lidar_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="lidar_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.1 0.1 0.1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>

      <sensor name="lidar_3d_sensor" type="ray">
        <ray>
          <scan>
            <horizontal>
              <samples>1080</samples>  <!-- High resolution for detailed scans -->
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle>  <!-- -π radians (-180°) -->
              <max_angle>3.14159</max_angle>   <!-- π radians (180°) -->
            </horizontal>
            <vertical>
              <samples>64</samples>    <!-- 64 beams for 3D scanning -->
              <resolution>1</resolution>
              <min_angle>-0.5236</min_angle>  <!-- -30° -->
              <max_angle>0.3491</max_angle>    <!-- 20° -->
            </vertical>
          </scan>
          <range>
            <min>0.1</min>      <!-- 10cm minimum range -->
            <max>100.0</max>    <!-- 100m maximum range -->
            <resolution>0.01</resolution>  <!-- 1cm resolution -->
          </range>
        </ray>

        <!-- Noise parameters for realistic simulation -->
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
          <dynamic_bias_stddev>0.0001</dynamic_bias_stddev>
          <dynamic_bias_correlation_time>1.0</dynamic_bias_correlation_time>
        </noise>

        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
      </sensor>
    </link>
  </model>
</sdf>
```

#### 2. LiDAR Noise Models
Different LiDAR models have different noise characteristics:

```xml
<!-- Velodyne VLP-16 style LiDAR -->
<sensor name="velodyne_vlp16" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>512</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.2618</min_angle>  <!-- -15° -->
        <max_angle>0.2618</max_angle>   <!-- 15° -->
      </vertical>
    </scan>
    <range>
      <min>0.2</min>
      <max>120.0</max>
      <resolution>0.004</resolution>
    </range>
  </ray>

  <!-- Specific noise model for VLP-16 -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.02</stddev>  <!-- 2cm accuracy -->
    <bias_mean>0.0</bias_mean>
    <bias_stddev>0.002</bias_stddev>
  </noise>
</sensor>

<!-- Hokuyo UTM-30LX style LiDAR -->
<sensor name="hokuyo_utm30lx" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>1081</samples>  <!-- 1081 points for 270° scan -->
        <resolution>1</resolution>
        <min_angle>-2.3562</min_angle>  <!-- -135° -->
        <max_angle>2.3562</max_angle>   <!-- 135° -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>

  <!-- Specific noise model for Hokuyo -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.015</stddev>  <!-- 1.5cm accuracy -->
    <bias_mean>0.0</bias_mean>
    <bias_stddev>0.0015</bias_stddev>
  </noise>
</sensor>
```

### RGB-D Camera Simulation

#### 1. Creating a Realistic RGB-D Camera Model
Create `models/rgbd_camera/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="rgbd_camera">
    <link name="camera_link">
      <pose>0 0 0 0 0 0</pose>

      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.0001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.0001</iyy>
          <iyz>0.0</iyz>
          <izz>0.0001</izz>
        </inertia>
      </inertial>

      <collision name="camera_collision">
        <geometry>
          <box>
            <size>0.05 0.05 0.03</size>
          </box>
        </geometry>
      </collision>

      <visual name="camera_visual">
        <geometry>
          <box>
            <size>0.05 0.05 0.03</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>

      <!-- RGB Camera -->
      <sensor name="rgb_camera" type="camera">
        <camera>
          <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10</far>
          </clip>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.007</stddev>  <!-- Noise level for RGB channel -->
          </noise>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
      </sensor>

      <!-- Depth Camera -->
      <sensor name="depth_camera" type="depth_camera">
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>L8</format>  <!-- 8-bit grayscale for depth -->
          </image>
          <clip>
            <near>0.1</near>
            <far>10</far>
          </clip>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.01</stddev>  <!-- 1cm depth accuracy -->
          </noise>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>false</visualize>
      </sensor>
    </link>
  </model>
</sdf>
```

#### 2. Advanced RGB-D Camera with Distortion
```xml
<!-- Realistic RGB-D camera with lens distortion -->
<sensor name="realistic_rgbd_camera" type="rgbd_camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
    <distortion>
      <k1>-0.22</k1>      <!-- Radial distortion coefficient -->
      <k2>0.24</k2>       <!-- Radial distortion coefficient -->
      <k3>0.0</k3>        <!-- Radial distortion coefficient -->
      <p1>0.0001</p1>     <!-- Tangential distortion coefficient -->
      <p2>0.0001</p2>     <!-- Tangential distortion coefficient -->
      <center>0.5 0.5</center>  <!-- Principal point (normalized) -->
    </distortion>
  </camera>

  <!-- Noise models for RGB and depth channels -->
  <rgb_camera>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.005</stddev>
    </noise>
  </rgb_camera>

  <depth_camera>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.015</stddev>  <!-- 1.5cm depth noise -->
      <bias_mean>0.0</bias_mean>
      <bias_stddev>0.005</bias_stddev>
    </noise>
  </depth_camera>

  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensor Simulation

#### 1. Creating a Realistic IMU Model
Create `models/imu_sensor/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="imu_sensor">
    <link name="imu_link">
      <pose>0 0 0 0 0 0</pose>

      <inertial>
        <mass>0.05</mass>
        <inertia>
          <ixx>0.00001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.00001</iyy>
          <iyz>0.0</iyz>
          <izz>0.00001</izz>
        </inertia>
      </inertial>

      <collision name="imu_collision">
        <geometry>
          <box>
            <size>0.02 0.02 0.01</size>
          </box>
        </geometry>
      </collision>

      <visual name="imu_visual">
        <geometry>
          <box>
            <size>0.02 0.02 0.01</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>

      <sensor name="imu_sensor" type="imu">
        <always_on>1</always_on>
        <update_rate>100</update_rate>
        <imu>
          <!-- Linear Accelerometer Noise -->
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>  <!-- ~1.7 mg RMS noise -->
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.001</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.001</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.001</bias_stddev>
              </noise>
            </z>
          </linear_acceleration>

          <!-- Angular Velocity (Gyroscope) Noise -->
          <angular_velocity>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>  <!-- ~0.057 deg/s RMS noise -->
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
                <dynamic_bias_correlation_time>300</dynamic_bias_correlation_time>
                <dynamic_bias_stddev>0.0001</dynamic_bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
                <dynamic_bias_correlation_time>300</dynamic_bias_correlation_time>
                <dynamic_bias_stddev>0.0001</dynamic_bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
                <dynamic_bias_correlation_time>300</dynamic_bias_correlation_time>
                <dynamic_bias_stddev>0.0001</dynamic_bias_stddev>
              </noise>
            </z>
          </angular_velocity>

          <!-- Orientation Noise (if orientation is provided) -->
          <orientation>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
              </noise>
            </z>
          </orientation>
        </imu>
        <visualize>false</visualize>
      </sensor>
    </link>
  </model>
</sdf>
```

#### 2. Different IMU Types with Specific Characteristics
```xml
<!-- High-accuracy IMU (e.g., tactical grade) -->
<sensor name="high_accuracy_imu" type="imu">
  <update_rate>1000</update_rate>
  <imu>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0001</stddev>  <!-- Very low noise -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.00001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0001</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.00001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0001</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.00001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00001</stddev>  <!-- Very low noise -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
          <dynamic_bias_correlation_time>1000</dynamic_bias_correlation_time>
          <dynamic_bias_stddev>0.000001</dynamic_bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00001</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
          <dynamic_bias_correlation_time>1000</dynamic_bias_correlation_time>
          <dynamic_bias_stddev>0.000001</dynamic_bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00001</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
          <dynamic_bias_correlation_time>1000</dynamic_bias_correlation_time>
          <dynamic_bias_stddev>0.000001</dynamic_bias_stddev>
        </noise>
      </z>
    </angular_velocity>
  </imu>
</sensor>

<!-- Consumer-grade IMU (e.g., smartphone) -->
<sensor name="consumer_imu" type="imu">
  <update_rate>100</update_rate>
  <imu>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>  <!-- Higher noise -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.01</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.01</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.01</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>  <!-- Higher noise -->
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
          <dynamic_bias_correlation_time>100</dynamic_bias_correlation_time>
          <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
          <dynamic_bias_correlation_time>100</dynamic_bias_correlation_time>
          <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.001</bias_stddev>
          <dynamic_bias_correlation_time>100</dynamic_bias_correlation_time>
          <dynamic_bias_stddev>0.001</dynamic_bias_stddev>
        </noise>
      </z>
    </angular_velocity>
  </imu>
</sensor>
```

### Sensor Fusion and Integration

#### 1. Multi-Sensor Integration Example
```xml
<!-- Robot with multiple sensors -->
<model name="sensor_equipped_robot">
  <!-- LiDAR on top -->
  <link name="lidar_mount">
    <pose>0 0 0.5 0 0 0</pose>
    <sensor name="main_lidar" type="ray">
      <!-- LiDAR configuration -->
    </sensor>
  </link>

  <!-- RGB-D camera on front -->
  <link name="camera_mount">
    <pose>0.2 0 0.3 0 0 0</pose>
    <sensor name="front_camera" type="rgbd_camera">
      <!-- Camera configuration -->
    </sensor>
  </link>

  <!-- IMU in center of mass -->
  <link name="imu_mount">
    <pose>0 0 0.1 0 0 0</pose>
    <sensor name="body_imu" type="imu">
      <!-- IMU configuration -->
    </sensor>
  </link>

  <!-- Additional sensors -->
  <sensor name="floor_contact_sensor" type="contact">
    <contact>
      <collision>base_collision</collision>
    </contact>
    <always_on>1</always_on>
    <update_rate>100</update_rate>
  </sensor>
</model>
```

### ROS 2 Sensor Integration

#### 1. Sensor Plugin Configuration
```xml
<!-- Add ROS 2 plugins for sensor data publishing -->
<sensor name="lidar_3d_sensor" type="ray">
  <!-- Sensor configuration as before -->

  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>robot1</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>robot1/lidar_link</frame_name>
    <update_rate>10</update_rate>
  </plugin>
</sensor>

<sensor name="rgb_camera" type="camera">
  <!-- Camera configuration as before -->

  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>robot1</namespace>
      <remapping>~/image_raw:=camera/image_raw</remapping>
      <remapping>~/camera_info:=camera/camera_info</remapping>
    </ros>
    <camera_name>front_camera</camera_name>
    <frame_name>robot1/camera_link</frame_name>
    <update_rate>30</update_rate>
    <hack_baseline>0.07</hack_baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
  </plugin>
</sensor>

<sensor name="imu_sensor" type="imu">
  <!-- IMU configuration as before -->

  <plugin name="imu_controller" filename="libgazebo_ros_imu_sensor.so">
    <ros>
      <namespace>robot1</namespace>
      <remapping>~/out:=imu/data</remapping>
    </ros>
    <frame_name>robot1/imu_link</frame_name>
    <initial_orientation_as_reference>false</initial_orientation_as_reference>
  </plugin>
</sensor>
```

### Sensor Validation and Calibration

#### 1. Python Script for Sensor Data Analysis
Create `scripts/sensor_analysis.py`:

```python
#!/usr/bin/env python3

"""
Sensor data analysis and validation script.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import statistics
from collections import deque
import matplotlib.pyplot as plt


class SensorAnalyzer(Node):
    """
    Analyze sensor data to validate simulation realism.
    """

    def __init__(self):
        super().__init__('sensor_analyzer')

        # Initialize data collectors
        self.scan_buffer = deque(maxlen=100)
        self.imu_buffer = deque(maxlen=1000)
        self.bridge = CvBridge()

        # Subscribe to sensor topics
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/robot1/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/robot1/imu/data',
            self.imu_callback,
            10
        )

        # Timer for analysis
        self.analysis_timer = self.create_timer(1.0, self.perform_analysis)

        self.get_logger().info('Sensor analyzer initialized')

    def scan_callback(self, msg):
        """Process incoming LiDAR data."""
        self.scan_buffer.append({
            'ranges': msg.ranges,
            'intensities': msg.intensities,
            'time_increment': msg.time_increment,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        })

    def imu_callback(self, msg):
        """Process incoming IMU data."""
        self.imu_buffer.append({
            'linear_acceleration': [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ],
            'angular_velocity': [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ],
            'orientation': [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ],
            'header': msg.header
        })

    def perform_analysis(self):
        """Perform statistical analysis on collected sensor data."""
        if len(self.scan_buffer) > 0:
            self.analyze_lidar_data()

        if len(self.imu_buffer) > 0:
            self.analyze_imu_data()

    def analyze_lidar_data(self):
        """Analyze LiDAR data statistics."""
        if len(self.scan_buffer) == 0:
            return

        # Collect range data
        all_ranges = []
        for scan_data in self.scan_buffer:
            valid_ranges = [r for r in scan_data['ranges'] if 0.1 < r < 10.0]  # Filter valid ranges
            all_ranges.extend(valid_ranges)

        if len(all_ranges) == 0:
            return

        # Calculate statistics
        mean_range = statistics.mean(all_ranges)
        stdev_range = statistics.stdev(all_ranges) if len(all_ranges) > 1 else 0.0
        min_range = min(all_ranges)
        max_range = max(all_ranges)

        self.get_logger().info(
            f'LiDAR Analysis - Mean: {mean_range:.3f}m, '
            f'StdDev: {stdev_range:.3f}m, '
            f'Min: {min_range:.3f}m, Max: {max_range:.3f}m'
        )

        # Check for realistic noise levels
        if stdev_range > 0.05:  # If noise is too high
            self.get_logger().warn('LiDAR noise level may be too high')

    def analyze_imu_data(self):
        """Analyze IMU data statistics."""
        if len(self.imu_buffer) < 10:  # Need enough samples for meaningful statistics
            return

        # Separate acceleration and gyroscope data
        acc_x = [sample['linear_acceleration'][0] for sample in self.imu_buffer]
        acc_y = [sample['linear_acceleration'][1] for sample in self.imu_buffer]
        acc_z = [sample['linear_acceleration'][2] for sample in self.imu_buffer]

        gyro_x = [sample['angular_velocity'][0] for sample in self.imu_buffer]
        gyro_y = [sample['angular_velocity'][1] for sample in self.imu_buffer]
        gyro_z = [sample['angular_velocity'][2] for sample in self.imu_buffer]

        # Calculate statistics for accelerometer
        acc_stats = {
            'x': {'mean': statistics.mean(acc_x), 'stdev': statistics.stdev(acc_x) if len(acc_x) > 1 else 0.0},
            'y': {'mean': statistics.mean(acc_y), 'stdev': statistics.stdev(acc_y) if len(acc_y) > 1 else 0.0},
            'z': {'mean': statistics.mean(acc_z), 'stdev': statistics.stdev(acc_z) if len(acc_z) > 1 else 0.0}
        }

        # Calculate statistics for gyroscope
        gyro_stats = {
            'x': {'mean': statistics.mean(gyro_x), 'stdev': statistics.stdev(gyro_x) if len(gyro_x) > 1 else 0.0},
            'y': {'mean': statistics.mean(gyro_y), 'stdev': statistics.stdev(gyro_y) if len(gyro_y) > 1 else 0.0},
            'z': {'mean': statistics.mean(gyro_z), 'stdev': statistics.stdev(gyro_z) if len(gyro_z) > 1 else 0.0}
        }

        self.get_logger().info(
            f'IMU Acc Stats - X: μ={acc_stats["x"]["mean"]:.3f}, σ={acc_stats["x"]["stdev"]:.3f} | '
            f'Y: μ={acc_stats["y"]["mean"]:.3f}, σ={acc_stats["y"]["stdev"]:.3f} | '
            f'Z: μ={acc_stats["z"]["mean"]:.3f}, σ={acc_stats["z"]["stdev"]:.3f}'
        )

        self.get_logger().info(
            f'IMU Gyro Stats - X: μ={gyro_stats["x"]["mean"]:.3f}, σ={gyro_stats["x"]["stdev"]:.3f} | '
            f'Y: μ={gyro_stats["y"]["mean"]:.3f}, σ={gyro_stats["y"]["stdev"]:.3f} | '
            f'Z: μ={gyro_stats["z"]["mean"]:.3f}, σ={gyro_stats["z"]["stdev"]:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)

    analyzer = SensorAnalyzer()

    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercises

1. Create a LiDAR sensor model with realistic noise characteristics matching a specific commercial sensor
2. Implement an RGB-D camera with lens distortion parameters based on real camera specifications
3. Configure an IMU with appropriate noise and bias parameters for a consumer-grade sensor
4. Integrate multiple sensor types on a robot model and verify proper data publication
5. Write a validation script to analyze sensor data statistics and verify realism
6. Compare simulated sensor data with real sensor data to validate the simulation
7. Create a sensor fusion node that combines data from multiple simulated sensors

## References

1. Gazebo Sensor Documentation: http://gazebosim.org/tutorials?tut=sensor_noise
2. ROS 2 Gazebo Sensor Plugins: http://gazebosim.org/tutorials/?tut=ros_gz_plugins
3. Sensor Noise Modeling: https://www.mdpi.com/1424-8220/18/12/4423
4. IMU Modeling Techniques: https://www.sciencedirect.com/science/article/pii/S0921889018303305

## Further Reading

- Advanced sensor fusion techniques for robotics
- Dynamic sensor calibration in simulation
- Transfer learning from simulation to real robots
- Multi-robot sensor networks in simulated environments