---
sidebar_position: 34
---

# Nav2 Configuration for Bipedal Path Planning

## Learning Objectives
By the end of this module, students will be able to:
- Configure Nav2 for bipedal robot navigation with dynamic stability constraints
- Implement custom path planners suitable for humanoid locomotion
- Integrate balance-aware navigation with gait pattern generation
- Tune navigation parameters for bipedal locomotion safety
- Validate navigation performance in simulated and real environments

## Theory

### Bipedal Navigation Challenges

Bipedal navigation presents unique challenges compared to wheeled or tracked robots:

#### 1. Dynamic Stability Requirements
- Bipedal robots require continuous balance maintenance
- Center of Mass (CoM) must remain within support polygon
- Gait patterns must account for dynamic stability
- Footstep planning is critical for safe navigation

#### 2. Kinematic Constraints
- Limited step length and height
- Turning requires coordinated stepping patterns
- Obstacle clearance requires precise foot placement
- Slope navigation requires specialized gait patterns

#### 3. Computational Requirements
- Real-time balance control during navigation
- High-frequency sensor feedback processing
- Predictive control for dynamic stability
- Integration of multiple control systems (balance, navigation, perception)

### Navigation Stack Architecture for Bipedal Robots

#### 1. Layered Architecture
```
┌─────────────────────────────────┐
│         Task Planner            │  ← High-level mission planning
├─────────────────────────────────┤
│        Behavior Tree            │  ← Action selection & coordination
├─────────────────────────────────┤
│         Global Planner          │  ← Path planning with bipedal constraints
├─────────────────────────────────┤
│        Local Planner            │  ← Dynamic obstacle avoidance
├─────────────────────────────────┤
│         Footstep Planner        │  ← Safe foot placement
├─────────────────────────────────┤
│         Balance Controller      │  ← Dynamic balance maintenance
├─────────────────────────────────┤
│         Motor Controllers       │  ← Joint-level control
└─────────────────────────────────┘
```

#### 2. Bipedal-Specific Considerations
- **Footstep Planning**: Generate stable footstep sequences
- **ZMP (Zero Moment Point)**: Maintain balance during locomotion
- **Gait Pattern Generation**: Create stable walking gaits
- **Dynamic Stability**: Real-time balance control during navigation

### Key Concepts

#### 1. Footstep Planning
Footstep planning is crucial for bipedal navigation:
- Determines safe and stable foot placement locations
- Considers terrain traversability and robot kinematics
- Accounts for balance constraints during stepping
- Generates stepping patterns for obstacle negotiation

#### 2. ZMP-Based Navigation
Zero Moment Point (ZMP) is critical for dynamic stability:
- ZMP must remain within support polygon for stable walking
- Navigation paths must account for ZMP constraints
- Gait patterns are planned to maintain ZMP within bounds
- Real-time ZMP monitoring during navigation

#### 3. Balance-Aware Path Planning
Traditional path planning doesn't account for bipedal constraints:
- Step length limitations
- Turning radius based on stepping patterns
- Terrain slope limitations
- Obstacle height considerations for stepping

## Implementation

### Prerequisites
- ROS 2 Humble with Navigation2 packages
- Bipedal robot model with proper URDF
- Working VSLAM or localization system
- RTX workstation for simulation

### 1. Bipedal Robot Configuration

#### Robot Configuration File (bipedal_robot.yaml)
```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: False
    global_frame_id: "map"
    lambda_short: 0.1
    likelihood_max_dist: 2.0
    set_initial_pose: True
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0
    tf_broadcast: True
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_raycast_client:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_footprint
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    # Specify the path where the BT XML files are located
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_assisted_teleop_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # DWB Controller (modified for bipedal constraints)
    FollowPath:
      plugin: "nav2_mppi_controller::Controller"
      time_steps: 32
      control_freq: 20.0
      horizon_dt: 0.5
      reference_speed: 0.3  # Reduced for bipedal stability
      xy_resolution: 0.05
      yaw_resolution: 0.1
      motion_model: "Ackermann"
      weight_xy: 1.0
      weight_yaw: 0.05
      weight_vel: 0.0
      weight_cost: 1.0
      weight_smoothness: 0.2
      weight_obstacle: 0.5
      weight_dynamic_obstacle: 0.2
      weight_feasibility: 0.1
      weight_balance: 5.0  # High weight for balance constraint
      max_linear_speed: 0.4  # Reduced for bipedal safety
      min_linear_speed: 0.0
      max_angular_speed: 0.5
      min_angular_speed: 0.0
      max_accel: 0.2  # Conservative acceleration for bipedal
      max_decel: 0.3
      threshold_to_consider_goal_reached: 0.25
      threshold_y_to_consider_goal_reached: 0.25
      threshold_theta_to_consider_goal_reached: 0.2
      transform_tolerance: 0.1
      use_vel_scaled_traj: true
      use_interpolation: true
      use_global_plan_overrides: true
      use_cost_regulated_positive_only: false
      interpolation_behind_robot: false

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_footprint
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05  # High resolution for precise footstep planning
      robot_radius: 0.3  # Conservative radius for bipedal safety
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True

local_costmap_client:
  ros__parameters:
    use_sim_time: True

local_costmap_rclcpp_node:
  ros__parameters:
    use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_footprint
      use_sim_time: True
      robot_radius: 0.3  # Conservative for bipedal safety
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True

global_costmap_client:
  ros__parameters:
    use_sim_time: True

global_costmap_rclcpp_node:
  ros__parameters:
    use_sim_time: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      # Bipedal-specific parameters
      max_step_length: 0.3  # Maximum step length for bipedal
      min_step_length: 0.05 # Minimum step length
      max_turn_radius: 0.5  # Maximum turn radius based on step pattern
      step_height_threshold: 0.1  # Maximum step height for obstacle clearance

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

smoother_server:
  ros__parameters:
    use_sim_time: True
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1e-10
      max_its: 1000
      w_smooth: 0.3
      w_data: 0.2
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "assisted_teleop", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
      server_timeout: 20
      spins_addup: true
      max_spins: 1  # Conservative for bipedal stability
      spin_dist: 1.57  # 90 degrees in radians
    backup:
      plugin: "nav2_behaviors::BackUp"
      server_timeout: 20
      # Bipedal-specific backup parameters
      backup_dist: 0.15  # Conservative backup distance
      backup_speed: 0.05  # Slow backup speed for stability
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
      server_timeout: 20
      min_approach_dist: 0.1
      approach_angle_tol: 0.785  # 45 degrees
      drive_angle_tol: 0.785
      max_drive_dist: 0.3  # Conservative drive distance
      min_drive_dist: 0.05
      drive_speed: 0.1  # Slow speed for stability
    wait:
      plugin: "nav2_behaviors::Wait"
      server_timeout: 20
      sleep_interval: 0.1
      wait_sleep_interval: 0.1
    assisted_teleop:
      plugin: "nav2_behaviors::AssistedTeleop"
      server_timeout: 20
      min_vel_trans: 0.05
      max_vel_trans: 0.2  # Conservative linear velocity
      min_vel_rot: 0.05
      max_vel_rot: 0.2   # Conservative angular velocity
      min_bound_v: -0.1
      max_bound_v: 0.1
      min_bound_w: -0.1
      max_bound_w: 0.1

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 200  # Longer pause for bipedal stability
```

### 2. Bipedal-Specific Navigation Parameters

#### Custom Footstep Planner Configuration
Create `footstep_planner_params.yaml`:

```yaml
footstep_planner:
  ros__parameters:
    # Robot-specific parameters
    step_length_max: 0.3      # Maximum step length (m)
    step_length_min: 0.05     # Minimum step length (m)
    step_width_max: 0.25      # Maximum step width (m)
    step_height_max: 0.1      # Maximum step height (m)
    step_turn_max: 0.52       # Maximum turn per step (rad, ~30 degrees)

    # Foot dimensions
    foot_size_x: 0.15         # Foot length (m)
    foot_size_y: 0.08         # Foot width (m)

    # Balance parameters
    com_height: 0.7           # Center of mass height (m)
    zmp_margin: 0.05          # Safety margin for ZMP (m)
    support_polygon_ratio: 0.8 # Ratio of support polygon to use

    # Planning parameters
    planning_horizon: 10      # Number of steps to plan ahead
    replanning_frequency: 10  # Replanning frequency (Hz)
    step_timing: 0.5          # Time per step (s)

    # Terrain analysis
    max_slope: 0.3            # Maximum traversable slope (rad)
    max_step_up: 0.1          # Maximum step-up height (m)
    max_step_down: 0.15       # Maximum step-down height (m)

    # Obstacle avoidance
    min_obstacle_distance: 0.2 # Minimum distance to obstacles (m)
    foot_lift_height: 0.05    # Height to lift foot during step (m)

    # Cost weights
    cost_weights:
      path_deviation: 1.0     # Cost for deviating from path
      step_length: 0.5        # Cost for step length
      step_rotation: 0.3      # Cost for step rotation
      terrain_cost: 2.0       # Cost for terrain difficulty
      balance_cost: 3.0       # Cost for balance maintenance
      obstacle_cost: 5.0      # Cost for proximity to obstacles
```

#### 3. Balance-Aware Controller
Create `balance_controller.py`:

```python
#!/usr/bin/env python3
"""
Balance-aware controller for bipedal navigation.
Integrates with Nav2 to ensure dynamic stability during navigation.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
from builtin_interfaces.msg import Duration
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate

class BalanceAwareController(Node):
    """
    Balance-aware controller that integrates with Nav2 for bipedal navigation.
    """

    def __init__(self):
        super().__init__('balance_aware_controller')

        # Parameters
        self.declare_parameter('com_height', 0.7)
        self.declare_parameter('zmp_margin', 0.05)
        self.declare_parameter('max_step_length', 0.3)
        self.declare_parameter('step_timing', 0.5)

        self.com_height = self.get_parameter('com_height').value
        self.zmp_margin = self.get_parameter('zmp_margin').value
        self.max_step_length = self.get_parameter('max_step_length').value
        self.step_timing = self.get_parameter('step_timing').value

        # State variables
        self.current_pose = None
        self.current_twist = None
        self.current_imu = None
        self.foot_positions = {'left': None, 'right': None}
        self.support_foot = 'left'  # Current support foot

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_balance', 10)
        self.zmp_pub = self.create_publisher(Point, '/balance/zmp', 10)
        self.com_pub = self.create_publisher(Point, '/balance/com', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # TF listener for foot positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Balance control timer
        self.balance_timer = self.create_timer(0.05, self.balance_control_callback)  # 20Hz

        # Navigation state
        self.nav_active = False
        self.navigation_goal = None
        self.balance_threshold = 0.1  # Maximum acceptable balance error (m)

        self.get_logger().info('Balance-aware controller initialized')

    def odom_callback(self, msg):
        """Update robot pose and twist from odometry."""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def imu_callback(self, msg):
        """Update IMU data for balance estimation."""
        self.current_imu = msg

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection."""
        # Process scan data for obstacle detection near feet
        # This helps in adjusting step placement
        pass

    def get_foot_positions(self):
        """Get current foot positions from TF tree."""
        try:
            # Get transforms for left and right feet
            left_foot_tf = self.tf_buffer.lookup_transform(
                'map', 'left_foot', rclpy.time.Time())
            right_foot_tf = self.tf_buffer.lookup_transform(
                'map', 'right_foot', rclpy.time.Time())

            # Convert transforms to positions
            self.foot_positions['left'] = Point()
            self.foot_positions['left'].x = left_foot_tf.transform.translation.x
            self.foot_positions['left'].y = left_foot_tf.transform.translation.y
            self.foot_positions['left'].z = left_foot_tf.transform.translation.z

            self.foot_positions['right'] = Point()
            self.foot_positions['right'].x = right_foot_tf.transform.translation.x
            self.foot_positions['right'].y = right_foot_tf.transform.translation.y
            self.foot_positions['right'].z = right_foot_tf.transform.translation.z

        except Exception as e:
            self.get_logger().warn(f'Could not get foot positions: {e}')

    def calculate_zmp(self):
        """Calculate Zero Moment Point based on current state."""
        if self.current_pose is None or self.current_imu is None:
            return None

        # Simplified ZMP calculation (in real implementation, this would be more complex)
        # ZMP_x = CoM_x - (CoM_z / g) * CoM_acc_x
        # ZMP_y = CoM_y - (CoM_z / g) * CoM_acc_y

        # Get CoM position (approximated from base position)
        com_x = self.current_pose.position.x
        com_y = self.current_pose.position.y
        com_z = self.com_height  # Fixed CoM height assumption

        # Get IMU linear acceleration
        accel_x = self.current_imu.linear_acceleration.x
        accel_y = self.current_imu.linear_acceleration.y

        # Calculate ZMP
        g = 9.81  # gravity
        zmp_x = com_x - (com_z / g) * accel_x
        zmp_y = com_y - (com_z / g) * accel_y

        zmp_point = Point()
        zmp_point.x = zmp_x
        zmp_point.y = zmp_y
        zmp_point.z = 0.0  # ZMP is on ground plane

        # Publish ZMP for visualization
        self.zmp_pub.publish(zmp_point)

        # Publish CoM for visualization
        com_point = Point()
        com_point.x = com_x
        com_point.y = com_y
        com_point.z = com_z
        self.com_pub.publish(com_point)

        return zmp_point

    def check_balance_feasibility(self, proposed_step):
        """
        Check if a proposed step maintains balance.

        Args:
            proposed_step: Proposed step position as (x, y, theta)

        Returns:
            bool: True if step maintains balance, False otherwise
        """
        zmp = self.calculate_zmp()
        if zmp is None:
            return False

        # Calculate support polygon based on stance foot
        if self.support_foot == 'left' and self.foot_positions['left'] is not None:
            stance_foot = self.foot_positions['left']
        elif self.support_foot == 'right' and self.foot_positions['right'] is not None:
            stance_foot = self.foot_positions['right']
        else:
            # If we don't know stance foot, use current position
            stance_foot = self.current_pose.position if self.current_pose else Point(x=0, y=0, z=0)

        # Calculate support polygon (simplified as rectangle around stance foot)
        # In real implementation, this would be more sophisticated
        foot_width = 0.08  # Approximate foot width
        foot_length = 0.15  # Approximate foot length

        support_min_x = stance_foot.x - foot_length/2
        support_max_x = stance_foot.x + foot_length/2
        support_min_y = stance_foot.y - foot_width/2
        support_max_y = stance_foot.y + foot_width/2

        # Check if ZMP is within support polygon with margin
        zmp_in_support = (
            support_min_x + self.zmp_margin <= zmp.x <= support_max_x - self.zmp_margin and
            support_min_y + self.zmp_margin <= zmp.y <= support_max_y - self.zmp_margin
        )

        # Also check proposed step feasibility
        step_x, step_y, step_theta = proposed_step
        step_feasible = (
            abs(step_x) <= self.max_step_length and
            abs(step_y) <= self.max_step_length * 0.5 and  # Smaller lateral steps
            abs(step_theta) <= math.radians(30)  # 30 degree max turn
        )

        return zmp_in_support and step_feasible

    def generate_footsteps(self, path):
        """
        Generate feasible footsteps from navigation path.

        Args:
            path: Navigation path as list of poses

        Returns:
            list: Feasible footsteps as (x, y, theta) tuples
        """
        footsteps = []

        if len(path) < 2:
            return footsteps

        # Start with current position
        start_x = self.current_pose.position.x if self.current_pose else 0.0
        start_y = self.current_pose.position.y if self.current_pose else 0.0

        # Calculate required footsteps
        for i in range(1, len(path)):
            target_x = path[i].pose.position.x
            target_y = path[i].pose.position.y

            # Calculate direction and distance
            dx = target_x - start_x
            dy = target_y - start_y
            dist = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)

            # Generate steps based on maximum step length
            steps_needed = int(math.ceil(dist / self.max_step_length))

            for step in range(steps_needed):
                # Interpolate position
                interp_factor = (step + 1) / steps_needed
                step_x = start_x + dx * interp_factor
                step_y = start_y + dy * interp_factor

                # Check balance feasibility
                proposed_step = (step_x, step_y, angle)
                if self.check_balance_feasibility(proposed_step):
                    footsteps.append(proposed_step)
                else:
                    # If step is not feasible, try alternative
                    self.get_logger().warn(f'Step {step} not feasible, adjusting...')
                    # In real implementation, find alternative step
                    pass

            start_x, start_y = target_x, target_y

        return footsteps

    def balance_control_callback(self):
        """Main balance control loop."""
        # Update foot positions
        self.get_foot_positions()

        # Calculate current ZMP
        zmp = self.calculate_zmp()

        if zmp is not None:
            # Check if balance is maintained
            balance_error = self.estimate_balance_error(zmp)

            if balance_error > self.balance_threshold:
                # Emergency balance recovery
                self.execute_balance_recovery()
            else:
                # Normal navigation with balance awareness
                self.execute_balanced_navigation()

    def estimate_balance_error(self, zmp):
        """Estimate balance error based on ZMP position."""
        # Simplified balance error estimation
        # In real implementation, this would consider support polygon
        if self.current_pose is None:
            return float('inf')

        # Distance from CoM projection to ZMP
        com_proj_x = self.current_pose.position.x
        com_proj_y = self.current_pose.position.y

        error = math.sqrt((zmp.x - com_proj_x)**2 + (zmp.y - com_proj_y)**2)
        return error

    def execute_balance_recovery(self):
        """Execute emergency balance recovery procedure."""
        self.get_logger().warn('Executing balance recovery!')

        # Stop movement
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

    def execute_balanced_navigation(self):
        """Execute navigation with balance constraints."""
        # This would interface with Nav2's local planner
        # to ensure balance-aware velocity commands
        pass

def main(args=None):
    rclpy.init(args=args)

    balance_controller = BalanceAwareController()

    try:
        rclpy.spin(balance_controller)
    except KeyboardInterrupt:
        pass
    finally:
        balance_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 4. Bipedal Navigation Launch File
Create `bipedal_nav2.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
import pathlib


def generate_launch_description():
    # Get path to package
    pkg_share = pathlib.Path(__file__).parent.parent.absolute()

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    bt_xml_file = LaunchConfiguration('bt_xml_file')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    # Declare launch arguments
    declare_use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation/Gazebo clock')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'bipedal_nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_bt_xml_cmd = DeclareLaunchArgument(
        'bt_xml_file',
        default_value=os.path.join(pkg_share, 'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
        description='Full path to the behavior tree xml file to use')

    declare_map_subscribe_transient_local_cmd = DeclareLaunchArgument(
        'map_subscribe_transient_local',
        default_value='false',
        description='Whether to set the map subscriber to transient local')

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'autostart': autostart,
        'map_subscribe_transient_local': map_subscribe_transient_local}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True)

    # Include the nav2 launch file
    nav2_launch_file_dir = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch')

    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_file_dir, 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': configured_params,
            'autostart': autostart,
            'bt_xml_file': bt_xml_file,
            'map_subscribe_transient_local': map_subscribe_transient_local}.items())

    # Include the localization launch file
    localization_launch_file_dir = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch')

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(localization_launch_file_dir, 'localization_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': configured_params,
            'map': os.path.join(pkg_share, 'maps', 'apartment_map.yaml')}.items())

    # Balance-aware controller node
    balance_controller_node = Node(
        package='bipedal_navigation',
        executable='balance_controller',
        name='balance_controller',
        parameters=[configured_params],
        remappings=[('/cmd_vel_balance', '/cmd_vel')],
        output='screen'
    )

    # Footstep planner node
    footstep_planner_node = Node(
        package='bipedal_navigation',
        executable='footstep_planner',
        name='footstep_planner',
        parameters=[configured_params],
        output='screen'
    )

    # Bipedal-specific path follower
    bipedal_controller_node = Node(
        package='bipedal_navigation',
        executable='bipedal_path_follower',
        name='bipedal_path_follower',
        parameters=[configured_params],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Declare launch options
    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_bt_xml_cmd)
    ld.add_action(declare_map_subscribe_transient_local_cmd)

    # Add nodes and launch files
    ld.add_action(nav2_bringup_launch)
    ld.add_action(localization_launch)
    ld.add_action(balance_controller_node)
    ld.add_action(footstep_planner_node)
    ld.add_action(bipedal_controller_node)

    return ld
```

#### 5. Gait Pattern Generator
Create `gait_pattern_generator.py`:

```python
#!/usr/bin/env python3
"""
Gait pattern generator for bipedal navigation.
Generates stable walking patterns for navigation tasks.
"""

import numpy as np
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GaitPatternGenerator(Node):
    """
    Generates gait patterns for bipedal navigation.
    """

    def __init__(self):
        super().__init__('gait_pattern_generator')

        # Parameters
        self.declare_parameter('step_height', 0.05)
        self.declare_parameter('step_length', 0.3)
        self.declare_parameter('step_duration', 1.0)
        self.declare_parameter('stance_duration', 0.3)
        self.declare_parameter('swing_duration', 0.4)

        self.step_height = self.get_parameter('step_height').value
        self.step_length = self.get_parameter('step_length').value
        self.step_duration = self.get_parameter('step_duration').value
        self.stance_duration = self.get_parameter('stance_duration').value
        self.swing_duration = self.get_parameter('swing_duration').value

        # Robot joint names (adjust based on your robot)
        self.joint_names = [
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/bipedal_controller/joint_trajectory', 10)

        self.command_pub = self.create_publisher(
            Twist, '/bipedal_cmd_vel', 10)

        # Subscribers
        self.nav_cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_nav', self.nav_command_callback, 10)

        # Gait state
        self.current_gait_phase = 0.0
        self.is_moving = False
        self.target_velocity = Twist()

        # Timer for gait generation
        self.gait_timer = self.create_timer(0.01, self.gait_generation_callback)

        self.get_logger().info('Gait pattern generator initialized')

    def nav_command_callback(self, msg):
        """Receive navigation commands and set target velocity."""
        self.target_velocity = msg
        self.is_moving = abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01

    def generate_walk_pattern(self, linear_vel, angular_vel, time_step):
        """
        Generate walking pattern based on desired velocity.

        Args:
            linear_vel: Desired linear velocity (m/s)
            angular_vel: Desired angular velocity (rad/s)
            time_step: Current time in gait cycle

        Returns:
            JointTrajectory: Trajectory message with joint positions
        """
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        # Calculate gait parameters based on velocity
        # Scale step length and frequency based on desired speed
        speed_factor = min(abs(linear_vel) / 0.3, 1.0)  # Max 0.3 m/s
        step_length = self.step_length * speed_factor
        step_frequency = 1.0 + speed_factor * 0.5  # 1-1.5 Hz

        # Generate foot trajectories for current gait phase
        left_foot_pos, right_foot_pos = self.calculate_foot_trajectories(
            time_step, step_length, step_frequency, angular_vel)

        # Convert foot positions to joint angles (simplified)
        joint_positions = self.inverse_kinematics_batch(
            left_foot_pos, right_foot_pos)

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_positions)  # Simplified
        point.accelerations = [0.0] * len(joint_positions)  # Simplified
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(10000000)  # 10ms

        trajectory_msg.points = [point]
        return trajectory_msg

    def calculate_foot_trajectories(self, time_step, step_length, step_frequency, angular_vel):
        """
        Calculate foot trajectories for walking gait.

        Args:
            time_step: Current time in gait cycle
            step_length: Desired step length
            step_frequency: Step frequency (Hz)
            angular_vel: Desired angular velocity

        Returns:
            tuple: (left_foot_position, right_foot_position)
        """
        # Simplified foot trajectory calculation
        # In real implementation, this would be more sophisticated

        # Calculate phase for each foot (alternating)
        left_phase = (time_step * step_frequency) % 2.0
        right_phase = ((time_step * step_frequency) + 1.0) % 2.0

        # Calculate foot positions based on phase
        left_foot = self.calculate_foot_position(left_phase, step_length, angular_vel)
        right_foot = self.calculate_foot_position(right_phase, step_length, angular_vel)

        return left_foot, right_foot

    def calculate_foot_position(self, phase, step_length, angular_vel):
        """
        Calculate single foot position based on gait phase.

        Args:
            phase: Current gait phase (0-2, where 0-1 is stance, 1-2 is swing)
            step_length: Step length
            angular_vel: Angular velocity for turning

        Returns:
            list: [x, y, z] foot position
        """
        if phase < 1.0:  # Stance phase
            # Foot is on ground, moving with robot
            x = 0.0  # Relative to hip
            y = 0.0
            z = 0.0  # On ground
        else:  # Swing phase
            # Foot is lifted and moved forward
            swing_phase = phase - 1.0  # Normalize to 0-1
            x = step_length * swing_phase  # Move forward
            y = angular_vel * 0.1 * (swing_phase - 0.5)  # Lateral adjustment for turning
            z = self.step_height * math.sin(math.pi * swing_phase)  # Lift foot

        return [x, y, z]

    def inverse_kinematics_batch(self, left_foot_pos, right_foot_pos):
        """
        Simplified inverse kinematics to convert foot positions to joint angles.
        In a real implementation, this would use proper IK solvers.

        Args:
            left_foot_pos: [x, y, z] position of left foot
            right_foot_pos: [x, y, z] position of right foot

        Returns:
            list: Joint positions for all joints
        """
        # This is a highly simplified IK calculation
        # In real implementation, use proper IK solvers like KDL or ikfast

        # Default joint positions (standing pose)
        joint_positions = [0.0] * 12  # 12 joints

        # Calculate hip and knee positions based on foot positions
        # Left leg
        joint_positions[0] = left_foot_pos[0] * 0.1  # Hip pitch
        joint_positions[1] = left_foot_pos[1] * 0.05  # Hip roll
        joint_positions[3] = min(0.5, max(-0.5, left_foot_pos[2] * 2.0))  # Knee

        # Right leg
        joint_positions[6] = right_foot_pos[0] * 0.1  # Hip pitch
        joint_positions[7] = right_foot_pos[1] * 0.05  # Hip roll
        joint_positions[9] = min(0.5, max(-0.5, right_foot_pos[2] * 2.0))  # Knee

        return joint_positions

    def gait_generation_callback(self):
        """Main gait generation loop."""
        if not self.is_moving:
            # Publish standing position when not moving
            standing_trajectory = self.generate_standing_trajectory()
            self.trajectory_pub.publish(standing_trajectory)
            return

        # Update gait phase
        self.current_gait_phase += 0.01  # 100Hz update
        if self.current_gait_phase > 1000:  # Prevent overflow
            self.current_gait_phase = 0.0

        # Generate walking pattern
        walk_trajectory = self.generate_walk_pattern(
            self.target_velocity.linear.x,
            self.target_velocity.angular.z,
            self.current_gait_phase
        )

        # Publish trajectory
        self.trajectory_pub.publish(walk_trajectory)

    def generate_standing_trajectory(self):
        """Generate trajectory for standing position."""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        # Standing position joint angles
        standing_positions = [0.0] * len(self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = standing_positions
        point.velocities = [0.0] * len(standing_positions)
        point.accelerations = [0.0] * len(standing_positions)
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(10000000)  # 10ms

        trajectory_msg.points = [point]
        return trajectory_msg


def main(args=None):
    rclpy.init(args=args)

    gait_generator = GaitPatternGenerator()

    try:
        rclpy.spin(gait_generator)
    except KeyboardInterrupt:
        pass
    finally:
        gait_generator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Performance Tuning and Validation

#### 1. Navigation Performance Metrics
```python
# navigation_performance_evaluator.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32
import numpy as np
from scipy.spatial.distance import cdist
import math

class NavigationPerformanceEvaluator(Node):
    """
    Evaluates navigation performance for bipedal robots.
    """

    def __init__(self):
        super().__init__('navigation_performance_evaluator')

        # Parameters
        self.declare_parameter('evaluation_window', 100)
        self.declare_parameter('success_radius', 0.5)
        self.declare_parameter('time_limit', 300.0)  # 5 minutes

        self.eval_window = self.get_parameter('evaluation_window').value
        self.success_radius = self.get_parameter('success_radius').value
        self.time_limit = self.get_parameter('time_limit').value

        # Subscribers
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)

        # Publishers
        self.performance_pub = self.create_publisher(
            Float32, '/navigation_performance', 10)
        self.metrics_pub = self.create_publisher(
            MarkerArray, '/navigation_metrics', 10)

        # State variables
        self.current_path = None
        self.current_pose = None
        self.navigation_start_time = None
        self.navigation_goal = None
        self.path_execution_log = []

        # Performance metrics
        self.metrics = {
            'path_efficiency': [],  # Actual distance / optimal distance
            'execution_time': [],   # Time to reach goal
            'success_rate': [],     # Success/failure
            'stability_score': [],  # Balance maintenance
            'obstacle_clearance': [] # Distance to obstacles
        }

        self.get_logger().info('Navigation performance evaluator initialized')

    def path_callback(self, msg):
        """Receive planned path."""
        self.current_path = msg.poses

    def odom_callback(self, msg):
        """Receive robot odometry."""
        self.current_pose = msg.pose.pose

        if self.navigation_start_time is not None and self.navigation_goal is not None:
            # Log current position for analysis
            self.path_execution_log.append({
                'timestamp': self.get_clock().now().seconds_nanoseconds(),
                'position': msg.pose.pose.position,
                'distance_to_goal': self.calculate_distance_to_goal(msg.pose.pose.position)
            })

            # Check if goal reached
            if self.calculate_distance_to_goal(msg.pose.pose.position) < self.success_radius:
                self.record_navigation_success()

    def goal_callback(self, msg):
        """Receive navigation goal."""
        self.navigation_goal = msg.pose.position
        self.navigation_start_time = self.get_clock().now().seconds_nanoseconds()
        self.path_execution_log = []  # Reset log for new goal

    def calculate_distance_to_goal(self, position):
        """Calculate distance from position to goal."""
        if self.navigation_goal is None:
            return float('inf')

        dx = position.x - self.navigation_goal.x
        dy = position.y - self.navigation_goal.y
        return math.sqrt(dx*dx + dy*dy)

    def record_navigation_success(self):
        """Record successful navigation."""
        if self.navigation_start_time is None:
            return

        # Calculate execution time
        end_time = self.get_clock().now().seconds_nanoseconds()
        execution_time = end_time[0] - self.navigation_start_time[0] + (end_time[1] - self.navigation_start_time[1]) * 1e-9

        # Calculate path efficiency (if we have the original path)
        optimal_distance = self.calculate_optimal_path_distance()
        actual_distance = self.calculate_actual_path_distance()

        if optimal_distance > 0:
            path_efficiency = actual_distance / optimal_distance
        else:
            path_efficiency = 1.0

        # Add to metrics
        self.metrics['path_efficiency'].append(path_efficiency)
        self.metrics['execution_time'].append(execution_time)
        self.metrics['success_rate'].append(1.0)  # Success

        self.get_logger().info(f'Navigation succeeded! Time: {execution_time:.2f}s, Efficiency: {path_efficiency:.2f}')

        # Reset for next navigation
        self.navigation_start_time = None
        self.navigation_goal = None

    def calculate_optimal_path_distance(self):
        """Calculate optimal path distance (straight line)."""
        if self.current_path and len(self.current_path) >= 2:
            start = self.current_path[0].pose.position
            end = self.current_path[-1].pose.position
            dx = end.x - start.x
            dy = end.y - start.y
            return math.sqrt(dx*dx + dy*dy)
        return 0.0

    def calculate_actual_path_distance(self):
        """Calculate actual path distance traveled."""
        if len(self.path_execution_log) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(self.path_execution_log)):
            prev_pos = self.path_execution_log[i-1]['position']
            curr_pos = self.path_execution_log[i]['position']

            dx = curr_pos.x - prev_pos.x
            dy = curr_pos.y - prev_pos.y
            total_distance += math.sqrt(dx*dx + dy*dy)

        return total_distance

    def evaluate_stability(self):
        """Evaluate navigation stability based on robot motion."""
        if len(self.path_execution_log) < 10:
            return 0.0  # Not enough data

        # Calculate smoothness of motion (low jerk indicates stability)
        velocities = []
        for i in range(1, len(self.path_execution_log)):
            pos1 = self.path_execution_log[i-1]['position']
            pos2 = self.path_execution_log[i]['position']

            dt = 0.1  # Assume 10Hz logging
            vel = math.sqrt((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2) / dt
            velocities.append(vel)

        # Calculate velocity variance (lower is more stable)
        if len(velocities) > 1:
            stability_score = 1.0 / (1.0 + np.var(velocities))  # Higher is better
        else:
            stability_score = 1.0

        return stability_score

    def publish_performance_metrics(self):
        """Publish current performance metrics."""
        if len(self.metrics['execution_time']) > 0:
            avg_time = np.mean(self.metrics['execution_time'])
            avg_efficiency = np.mean(self.metrics['path_efficiency'])
            success_rate = np.mean(self.metrics['success_rate']) if self.metrics['success_rate'] else 0.0

            performance_msg = Float32()
            # Combine metrics into a single performance score
            performance_score = (
                0.3 * (1.0 / max(avg_time, 1.0)) +  # Faster is better
                0.4 * min(avg_efficiency, 1.0) +    # Efficiency capped at 1
                0.3 * success_rate                   # Success rate
            ) * 100  # Scale to 0-100

            performance_msg.data = performance_score
            self.performance_pub.publish(performance_msg)

            self.get_logger().info(
                f'Performance - Avg Time: {avg_time:.2f}s, '
                f'Efficiency: {avg_efficiency:.2f}, '
                f'Success Rate: {success_rate:.2f}, '
                f'Score: {performance_score:.1f}'
            )

def main(args=None):
    rclpy.init(args=args)

    evaluator = NavigationPerformanceEvaluator()

    # Timer to periodically publish metrics
    timer = evaluator.create_timer(5.0, evaluator.publish_performance_metrics)

    try:
        rclpy.spin(evaluator)
    except KeyboardInterrupt:
        pass
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. Configure Nav2 for a bipedal robot model with custom parameters
2. Implement a footstep planner that generates stable stepping patterns
3. Create a balance-aware controller that integrates with Nav2
4. Develop gait pattern generators for different walking speeds and terrains
5. Tune navigation parameters for optimal bipedal locomotion performance
6. Validate navigation performance in simulated apartment environments
7. Test navigation robustness with dynamic obstacles
8. Implement recovery behaviors specific to bipedal robots

## References

1. Navigation2 Documentation: https://navigation.ros.org/
2. Bipedal Navigation Papers: https://arxiv.org/search/?query=bipedal+navigation
3. ROS 2 Navigation Tutorials: https://navigation.ros.org/tutorials/
4. Footstep Planning: https://github.com/roboskel/footstep_planner
5. Balance Control for Humanoids: https://arxiv.org/abs/2003.09111

## Further Reading

- Advanced path planning for humanoid robots
- Dynamic balance control during navigation
- Multi-modal navigation (walking, climbing stairs, etc.)
- Learning-based navigation for bipedal robots
- Integration with perception systems for semantic navigation
- Human-aware navigation for social robotics applications