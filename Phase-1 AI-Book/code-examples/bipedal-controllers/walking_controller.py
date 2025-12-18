#!/usr/bin/env python3

"""
Bipedal Walking Controller
Implements basic walking patterns for humanoid robots
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import math


class BipedalWalkingController(Node):
    def __init__(self):
        super().__init__('bipedal_walking_controller')

        # Publishers and subscribers
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Walking parameters
        self.step_height = 0.1  # meters
        self.step_length = 0.3  # meters
        self.step_duration = 2.0  # seconds
        self.current_phase = 0.0

        # Joint names
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Timer for walking pattern
        self.timer = self.create_timer(0.01, self.walk_callback)  # 100Hz

        self.get_logger().info('Bipedal Walking Controller initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z
        # Process velocity commands to adjust walking pattern
        self.get_logger().info(f'Received velocity command: {linear_vel}, {angular_vel}')

    def generate_walking_pattern(self, phase):
        """Generate walking pattern based on current phase"""
        # Simplified walking pattern
        t = phase

        # Left leg pattern
        left_hip = math.sin(t * math.pi) * 0.2
        left_knee = math.sin(t * math.pi + math.pi/2) * 0.1
        left_ankle = math.sin(t * math.pi + math.pi) * 0.05

        # Right leg pattern (opposite phase)
        right_hip = math.sin(t * math.pi + math.pi) * 0.2
        right_knee = math.sin(t * math.pi + 3*math.pi/2) * 0.1
        right_ankle = math.sin(t * math.pi + 2*math.pi) * 0.05

        return [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]

    def walk_callback(self):
        """Main walking control loop"""
        # Update phase
        self.current_phase += 0.01 * 2 * math.pi / self.step_duration
        if self.current_phase > 2 * math.pi:
            self.current_phase = 0.0

        # Generate walking pattern
        joint_positions = self.generate_walking_pattern(self.current_phase)

        # Publish joint commands
        msg = Float64MultiArray()
        msg.data = joint_positions
        self.joint_cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    controller = BipedalWalkingController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down bipedal controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()