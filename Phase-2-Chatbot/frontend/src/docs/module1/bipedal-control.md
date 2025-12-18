---
sidebar_position: 14
---

# ROS 2 Controller Tutorials for Bipedal Motion

## Learning Objectives
By the end of this module, students will be able to:
- Implement joint position controllers for bipedal locomotion
- Design gait patterns for stable walking
- Apply inverse kinematics for leg movement control
- Implement balance and stability control algorithms
- Tune controller parameters for optimal performance

## Theory

### Bipedal Locomotion Principles
Bipedal locomotion is the act of walking on two legs. In robotics, this presents unique challenges due to the dynamic balance requirements and the need for coordinated multi-joint control.

### Key Concepts
- **Zero Moment Point (ZMP)**: A criterion for dynamic balance in walking robots
- **Center of Mass (CoM)**: Critical for maintaining balance during locomotion
- **Gait Phases**: Double support (both feet on ground) and single support (one foot on ground)
- **Inverse Kinematics (IK)**: Calculating joint angles to achieve desired foot positions

### Walking Patterns
- **Static Walking**: Maintains balance at all times (slow, stable)
- **Dynamic Walking**: Uses momentum and balance recovery (faster, more human-like)
- **Limit Cycles**: Repetitive gait patterns that maintain stable locomotion

## Implementation

### Prerequisites
- ROS 2 Humble installed
- Basic understanding of URDF and robot modeling
- Knowledge of rclpy and ROS 2 concepts
- Understanding of basic control theory

### Setting Up Controller Infrastructure

#### 1. Installing ros2_control
```bash
sudo apt update
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
```

#### 2. Creating a Controller Configuration File
Create `config/bipedal_controller.yaml`:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_leg_controller:
      type: position_controllers/JointGroupPositionController

    right_leg_controller:
      type: position_controllers/JointGroupPositionController

    left_arm_controller:
      type: position_controllers/JointGroupPositionController

    right_arm_controller:
      type: position_controllers/JointGroupPositionController

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
      - left_ankle_joint

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_joint
      - right_knee_joint
      - right_ankle_joint

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_joint
      - left_elbow_joint
      - left_wrist_joint

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_joint
      - right_elbow_joint
      - right_wrist_joint
```

#### 3. Creating Controller Launch File
Create `launch/bipedal_control.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Load controller configuration from YAML file
    robot_controllers = PathJoinSubstitution(
        [FindPackageShare('your_robot_description'), 'config', 'bipedal_controller.yaml']
    )

    # Controller manager node
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_controllers, {'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
        output='both',
    )

    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='both',
    )

    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )

    # Leg controllers
    left_leg_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_leg_controller', '--controller-manager', '/controller_manager'],
    )

    right_leg_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_leg_controller', '--controller-manager', '/controller_manager'],
    )

    # Event handler to start controllers after joint broadcaster
    delay_left_leg_after_joint_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[left_leg_controller_spawner],
        )
    )

    delay_right_leg_after_joint_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=left_leg_controller_spawner,
            on_exit=[right_leg_controller_spawner],
        )
    )

    return LaunchDescription([
        control_node,
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        delay_left_leg_after_joint_broadcaster,
        delay_right_leg_after_joint_broadcaster,
    ])


from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
```

### Basic Position Controller Implementation

#### 1. Simple Joint Position Controller
Create `scripts/simple_position_controller.py`:

```python
#!/usr/bin/env python3

"""
Simple joint position controller for bipedal robot.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import math
import time


class SimplePositionController(Node):
    """
    Simple position controller for bipedal robot joints.
    """

    def __init__(self):
        super().__init__('simple_position_controller')

        # Publishers for joint positions
        self.left_leg_pub = self.create_publisher(
            Float64MultiArray,
            '/left_leg_controller/commands',
            10
        )
        self.right_leg_pub = self.create_publisher(
            Float64MultiArray,
            '/right_leg_controller/commands',
            10
        )

        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}

        # Walking parameters
        self.walk_phase = 0.0
        self.walk_frequency = 0.5  # Hz
        self.step_height = 0.05    # meters
        self.step_length = 0.1     # meters

        self.get_logger().info('Simple position controller initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state updates."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def control_loop(self):
        """Main control loop."""
        # Update walking phase
        self.walk_phase += 2 * math.pi * self.walk_frequency * 0.01  # dt = 0.01s
        if self.walk_phase > 2 * math.pi:
            self.walk_phase -= 2 * math.pi

        # Calculate desired joint positions for walking
        left_hip_pos, left_knee_pos, left_ankle_pos = self.calculate_leg_trajectory(
            'left', self.walk_phase
        )
        right_hip_pos, right_knee_pos, right_ankle_pos = self.calculate_leg_trajectory(
            'right', self.walk_phase + math.pi  # Opposite phase for right leg
        )

        # Publish commands
        left_cmd = Float64MultiArray()
        left_cmd.data = [left_hip_pos, left_knee_pos, left_ankle_pos]
        self.left_leg_pub.publish(left_cmd)

        right_cmd = Float64MultiArray()
        right_cmd.data = [right_hip_pos, right_knee_pos, right_ankle_pos]
        self.right_leg_pub.publish(right_cmd)

    def calculate_leg_trajectory(self, leg_side, phase):
        """Calculate joint positions for a walking trajectory."""
        # Simplified walking pattern
        # Hip joint: moves forward/backward
        hip_pos = self.step_length * math.sin(phase)

        # Knee joint: provides leg bending for step height
        knee_pos = self.step_height * (1 - math.cos(phase)) * 0.5

        # Ankle joint: maintains balance
        ankle_pos = -self.step_height * (1 - math.cos(phase)) * 0.25

        return hip_pos, knee_pos, ankle_pos


def main(args=None):
    rclpy.init(args=args)

    controller = SimplePositionController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Walking Controller with Inverse Kinematics

#### 1. Inverse Kinematics Solver
Create `scripts/ik_walking_controller.py`:

```python
#!/usr/bin/env python3

"""
Advanced walking controller using inverse kinematics for bipedal robot.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import math
import numpy as np


class IKWalkingController(Node):
    """
    Walking controller using inverse kinematics for precise foot placement.
    """

    def __init__(self):
        super().__init__('ik_walking_controller')

        # Publishers for joint positions
        self.left_leg_pub = self.create_publisher(
            Float64MultiArray,
            '/left_leg_controller/commands',
            10
        )
        self.right_leg_pub = self.create_publisher(
            Float64MultiArray,
            '/right_leg_controller/commands',
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Robot parameters (from URDF)
        self.upper_leg_length = 0.3  # meters
        self.lower_leg_length = 0.3  # meters

        # Walking parameters
        self.walk_phase = 0.0
        self.walk_frequency = 0.5
        self.step_height = 0.05
        self.step_length = 0.1
        self.com_height = 0.5  # Center of mass height

        # Walking state
        self.left_support = True  # Start with left leg support
        self.step_counter = 0

        self.get_logger().info('IK walking controller initialized')

    def control_loop(self):
        """Main control loop."""
        # Update walking phase
        self.walk_phase += 2 * math.pi * self.walk_frequency * 0.01
        if self.walk_phase > 2 * math.pi:
            self.walk_phase -= 2 * math.pi

        # Determine support leg and swing leg
        if self.left_support:
            support_leg = 'left'
            swing_leg = 'right'
        else:
            support_leg = 'right'
            swing_leg = 'left'

        # Calculate support leg position (stays on ground)
        support_pos = self.calculate_support_foot_position(support_leg)

        # Calculate swing leg trajectory
        swing_pos = self.calculate_swing_foot_trajectory(swing_leg, self.walk_phase)

        # Calculate joint angles using inverse kinematics
        if swing_leg == 'left':
            left_joints = self.inverse_kinematics_2d(
                swing_pos[0], swing_pos[1], swing_pos[2]
            )
            right_joints = self.inverse_kinematics_2d(
                support_pos[0], support_pos[1], support_pos[2]
            )
        else:
            left_joints = self.inverse_kinematics_2d(
                support_pos[0], support_pos[1], support_pos[2]
            )
            right_joints = self.inverse_kinematics_2d(
                swing_pos[0], swing_pos[1], swing_pos[2]
            )

        # Publish commands
        if left_joints is not None:
            left_cmd = Float64MultiArray()
            left_cmd.data = left_joints
            self.left_leg_pub.publish(left_cmd)

        if right_joints is not None:
            right_cmd = Float64MultiArray()
            right_cmd.data = right_joints
            self.right_leg_pub.publish(right_cmd)

        # Switch support leg every half cycle
        if self.walk_phase > math.pi:
            self.left_support = not self.left_support
            self.step_counter += 1

    def calculate_support_foot_position(self, leg_side):
        """Calculate position of support foot."""
        # Support foot stays on ground at nominal position
        if leg_side == 'left':
            x = 0.0  # Nominal position
            y = 0.08  # Offset to the side
        else:
            x = 0.0
            y = -0.08  # Offset to the side

        z = 0.0  # On ground
        return [x, y, z]

    def calculate_swing_foot_trajectory(self, leg_side, phase):
        """Calculate trajectory for swing foot."""
        # Phase-based trajectory: move from behind to in front
        x_offset = self.step_length * math.sin(phase)

        # Vertical trajectory: lift foot at beginning and end of phase
        if phase < math.pi:
            z_offset = self.step_height * math.sin(phase)  # Lift during first half
        else:
            z_offset = 0.0  # Keep on ground during second half

        # Side offset remains constant
        if leg_side == 'left':
            y_offset = 0.08
        else:
            y_offset = -0.08

        return [x_offset, y_offset, z_offset]

    def inverse_kinematics_2d(self, x, y, z):
        """
        2D inverse kinematics for leg with hip and knee joints.
        Calculates hip and knee angles to reach desired foot position.
        """
        try:
            # Calculate distance from hip to foot in x-z plane
            dist = math.sqrt(x**2 + z**2)

            # Check if position is reachable
            if dist > (self.upper_leg_length + self.lower_leg_length):
                self.get_logger().warn('Position not reachable')
                return None

            # Calculate knee angle using law of cosines
            cos_knee = (self.upper_leg_length**2 + self.lower_leg_length**2 - dist**2) / \
                       (2 * self.upper_leg_length * self.lower_leg_length)
            cos_knee = max(-1, min(1, cos_knee))  # Clamp to valid range
            knee_angle = math.pi - math.acos(cos_knee)

            # Calculate angle between upper leg and line to foot
            cos_upper = (self.upper_leg_length**2 + dist**2 - self.lower_leg_length**2) / \
                        (2 * self.upper_leg_length * dist)
            cos_upper = max(-1, min(1, cos_upper))  # Clamp to valid range
            alpha = math.acos(cos_upper)

            # Calculate hip angle
            beta = math.atan2(z, x)  # Angle to foot from horizontal
            hip_angle = beta + alpha

            # Ankle angle to maintain foot orientation (simplified)
            ankle_angle = -(hip_angle + knee_angle - math.pi/2)

            return [hip_angle, knee_angle, ankle_angle]

        except Exception as e:
            self.get_logger().error(f'IK calculation failed: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)

    controller = IKWalkingController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Balance Controller Implementation

#### 1. Center of Mass Stabilizer
Create `scripts/com_stabilizer.py`:

```python
#!/usr/bin/env python3

"""
Center of Mass stabilizer for bipedal robot balance control.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
import math


class COMStabilizer(Node):
    """
    Stabilizer that adjusts robot posture to maintain center of mass within support polygon.
    """

    def __init__(self):
        super().__init__('com_stabilizer')

        # Publishers for joint positions
        self.left_leg_pub = self.create_publisher(
            Float64MultiArray,
            '/left_leg_controller/commands',
            10
        )
        self.right_leg_pub = self.create_publisher(
            Float64MultiArray,
            '/right_leg_controller/commands',
            10
        )
        self.torso_pub = self.create_publisher(
            Float64MultiArray,
            '/torso_controller/commands',
            10
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Robot state
        self.joint_positions = {}

        # Balance parameters
        self.com_x_desired = 0.0
        self.com_y_desired = 0.0
        self.com_z_desired = 0.5  # Desired CoM height

        # PID gains for balance control
        self.kp_x = 2.0
        self.ki_x = 0.1
        self.kd_x = 0.5
        self.kp_y = 2.0
        self.ki_y = 0.1
        self.kd_y = 0.5

        # PID error terms
        self.error_x_prev = 0.0
        self.error_y_prev = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0

        self.get_logger().info('COM stabilizer initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state updates."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def control_loop(self):
        """Main balance control loop."""
        # Estimate current CoM position (simplified model)
        com_pos = self.estimate_com_position()

        # Calculate errors
        error_x = self.com_x_desired - com_pos[0]
        error_y = self.com_y_desired - com_pos[1]

        # Update PID terms
        self.integral_x += error_x * 0.01  # dt = 0.01s
        self.integral_y += error_y * 0.01

        derivative_x = (error_x - self.error_x_prev) / 0.01
        derivative_y = (error_y - self.error_y_prev) / 0.01

        # Calculate control outputs
        control_x = (self.kp_x * error_x +
                    self.ki_x * self.integral_x +
                    self.kd_x * derivative_x)

        control_y = (self.kp_y * error_y +
                    self.ki_y * self.integral_y +
                    self.kd_y * derivative_y)

        # Apply control to adjust joint positions
        self.adjust_posture(control_x, control_y)

        # Update previous errors
        self.error_x_prev = error_x
        self.error_y_prev = error_y

    def estimate_com_position(self):
        """Estimate center of mass position (simplified)."""
        # This is a simplified estimation
        # In a real implementation, this would use a full kinematic model
        x = 0.0
        y = 0.0
        z = self.com_z_desired  # Assume constant height for simplicity

        # Add contributions from joint positions
        if 'left_hip_joint' in self.joint_positions:
            x += 0.1 * self.joint_positions['left_hip_joint']
        if 'right_hip_joint' in self.joint_positions:
            x += 0.1 * self.joint_positions['right_hip_joint']
        if 'torso_joint' in self.joint_positions:
            y += 0.2 * self.joint_positions['torso_joint']

        return [x, y, z]

    def adjust_posture(self, control_x, control_y):
        """Adjust robot posture based on balance control."""
        # Calculate desired joint adjustments
        # This is a simplified approach - in reality, more complex inverse kinematics would be needed

        # Adjust hip joints to shift CoM
        left_hip_adjust = control_y * 0.5
        right_hip_adjust = -control_y * 0.5

        # Adjust torso for balance
        torso_adjust = control_x * 0.3

        # Publish adjustments
        left_cmd = Float64MultiArray()
        left_cmd.data = [left_hip_adjust, 0.0, 0.0]  # Only hip for now
        self.left_leg_pub.publish(left_cmd)

        right_cmd = Float64MultiArray()
        right_cmd.data = [right_hip_adjust, 0.0, 0.0]  # Only hip for now
        self.right_leg_pub.publish(right_cmd)

        torso_cmd = Float64MultiArray()
        torso_cmd.data = [torso_adjust]
        self.torso_pub.publish(torso_cmd)


def main(args=None):
    rclpy.init(args=args)

    stabilizer = COMStabilizer()

    try:
        rclpy.spin(stabilizer)
    except KeyboardInterrupt:
        pass
    finally:
        stabilizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Parameter Tuning and Optimization

#### 1. Parameter Tuning Script
Create `scripts/tune_walking_params.py`:

```python
#!/usr/bin/env python3

"""
Parameter tuning script for bipedal walking controllers.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.srv import SetParameters
import time


class ParameterTuner(Node):
    """
    Parameter tuner for walking controllers.
    """

    def __init__(self):
        super().__init__('parameter_tuner')

        # Create client for parameter setting
        self.param_client = self.create_client(
            SetParameters,
            '/walking_controller/set_parameters'
        )

        # Wait for service
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Parameter service not available, waiting again...')

        # Timer for parameter updates
        self.tune_timer = self.create_timer(5.0, self.tune_parameters)

        self.tune_step = 0
        self.get_logger().info('Parameter tuner initialized')

    def tune_parameters(self):
        """Tune parameters using iterative approach."""
        # Define parameter ranges to test
        step_lengths = [0.05, 0.08, 0.1, 0.12, 0.15]
        step_heights = [0.02, 0.03, 0.05, 0.07]
        frequencies = [0.3, 0.4, 0.5, 0.6]

        # Cycle through different parameter combinations
        if self.tune_step < len(step_lengths) * len(step_heights) * len(frequencies):
            # Calculate current parameter combination
            freq_idx = self.tune_step // (len(step_lengths) * len(step_heights))
            height_idx = (self.tune_step // len(step_lengths)) % len(step_heights)
            length_idx = self.tune_step % len(step_lengths)

            # Set parameters
            self.set_walking_parameters(
                step_lengths[length_idx],
                step_heights[height_idx],
                frequencies[freq_idx]
            )

            self.get_logger().info(
                f'Testing parameters - length: {step_lengths[length_idx]}, '
                f'height: {step_heights[height_idx]}, '
                f'frequency: {frequencies[freq_idx]}'
            )

            self.tune_step += 1
        else:
            self.get_logger().info('Parameter tuning complete')
            self.tune_timer.cancel()

    def set_walking_parameters(self, step_length, step_height, frequency):
        """Set walking parameters on the controller."""
        # Create parameter requests
        params = [
            Parameter(
                name='step_length',
                value=ParameterValue(type=3, double_value=step_length)
            ),
            Parameter(
                name='step_height',
                value=ParameterValue(type=3, double_value=step_height)
            ),
            Parameter(
                name='walk_frequency',
                value=ParameterValue(type=3, double_value=frequency)
            )
        ]

        # Send request
        future = self.param_client.call_async(
            SetParameters.Request(parameters=params)
        )
        future.add_done_callback(self.param_set_callback)

    def param_set_callback(self, future):
        """Handle parameter setting response."""
        try:
            response = future.result()
            for result in response.results:
                if not result.successful:
                    self.get_logger().error(f'Parameter set failed: {result.reason}')
        except Exception as e:
            self.get_logger().error(f'Parameter service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)

    tuner = ParameterTuner()

    try:
        rclpy.spin(tuner)
    except KeyboardInterrupt:
        pass
    finally:
        tuner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercises

1. Implement the simple position controller and test it with your humanoid model
2. Enhance the IK walking controller to include more realistic gait patterns
3. Add sensor feedback (IMU, force/torque sensors) to the balance controller
4. Implement a ZMP (Zero Moment Point) based balance controller
5. Create a walking pattern generator that can transition between different walking speeds
6. Add arm swinging motion synchronized with leg movement for better balance
7. Implement parameter auto-tuning using ROS 2 dynamic reconfigure

## References

1. ROS 2 Control Documentation: https://control.ros.org/
2. ros2_control Tutorials: https://ros-controls.github.io/control.ros.org/
3. Bipedal Locomotion in Robotics: https://www.sciencedirect.com/topics/engineering/bipedal-locomotion
4. Inverse Kinematics in Robotics: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf

## Further Reading

- Advanced control strategies for bipedal robots
- Machine learning approaches to gait generation
- Dynamic balance and stability in humanoid robots
- Integration with perception systems for adaptive walking
- Hardware-in-the-loop simulation for controller validation