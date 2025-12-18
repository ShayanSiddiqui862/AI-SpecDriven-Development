#!/usr/bin/env python3

"""
IMU Sensor Simulation
Implements realistic IMU sensor with bias, drift, and noise for humanoid robotics
"""

import numpy as np
import math
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion
import rclpy
from rclpy.node import Node


class ImuSimulator(Node):
    def __init__(self):
        super().__init__('imu_simulator')

        # Publisher for simulated IMU data
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # IMU parameters
        self.linear_acceleration_noise = 0.017  # m/s^2
        self.angular_velocity_noise = 0.0015    # rad/s
        self.orientation_noise = 0.001          # rad

        # IMU bias and drift parameters
        self.accel_bias = [0.001, -0.002, 0.003]  # Small constant bias
        self.gyro_bias = [0.0001, -0.0002, 0.0003]  # Small constant bias

        # Drift rates (very slow)
        self.accel_drift_rate = [0.00001, 0.00001, 0.00001]
        self.gyro_drift_rate = [0.000001, 0.000001, 0.000001]

        # Current drift values
        self.current_accel_drift = [0.0, 0.0, 0.0]
        self.current_gyro_drift = [0.0, 0.0, 0.0]

        # Timer for publishing data (100Hz)
        self.timer = self.create_timer(0.01, self.publish_imu)

        # Time tracking
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info('IMU Simulator initialized')

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(x=x, y=y, z=z, w=w)

    def add_noise(self, linear_acceleration, angular_velocity, orientation):
        """Add realistic noise to IMU readings"""
        # Add noise to linear acceleration
        noisy_accel = [
            linear_acceleration.x + np.random.normal(self.accel_bias[0] + self.current_accel_drift[0], self.linear_acceleration_noise),
            linear_acceleration.y + np.random.normal(self.accel_bias[1] + self.current_accel_drift[1], self.linear_acceleration_noise),
            linear_acceleration.z + np.random.normal(self.accel_bias[2] + self.current_accel_drift[2], self.linear_acceleration_noise)
        ]

        # Add noise to angular velocity
        noisy_gyro = [
            angular_velocity.x + np.random.normal(self.gyro_bias[0] + self.current_gyro_drift[0], self.angular_velocity_noise),
            angular_velocity.y + np.random.normal(self.gyro_bias[1] + self.current_gyro_drift[1], self.angular_velocity_noise),
            angular_velocity.z + np.random.normal(self.gyro_bias[2] + self.current_gyro_drift[2], self.angular_velocity_noise)
        ]

        # For orientation, we'll add noise to the angles before converting back to quaternion
        # This is a simplified approach - in reality, orientation noise is more complex
        noisy_orientation = orientation

        return (Vector3(x=noisy_accel[0], y=noisy_accel[1], z=noisy_accel[2]),
                Vector3(x=noisy_gyro[0], y=noisy_gyro[1], z=noisy_gyro[2]),
                noisy_orientation)

    def simulate_humanoid_motion(self):
        """Simulate realistic humanoid motion patterns"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time

        # Simulate small oscillations that might occur during humanoid balance
        # These are typical of a walking or standing humanoid robot
        oscillation_freq = 0.5  # Hz - slow oscillations

        # Simulate small orientation changes (humanoid swaying)
        roll = 0.01 * math.sin(2 * math.pi * oscillation_freq * elapsed_time)
        pitch = 0.015 * math.cos(2 * math.pi * oscillation_freq * elapsed_time)
        yaw = 0.005 * math.sin(2 * math.pi * 0.2 * elapsed_time)  # Slow yaw changes

        # Simulate linear accelerations due to walking/balance
        # These are small accelerations due to the robot's movement
        linear_accel = Vector3()
        linear_accel.x = 0.1 * math.sin(2 * math.pi * 1.5 * elapsed_time)  # Forward/back motion
        linear_accel.y = 0.05 * math.cos(2 * math.pi * 1.2 * elapsed_time)  # Side motion
        linear_accel.z = 9.81 + 0.2 * math.sin(2 * math.pi * 2.0 * elapsed_time)  # Gravity + small vertical motion

        # Simulate angular velocities (slow changes in orientation)
        angular_vel = Vector3()
        angular_vel.x = 0.01 * math.cos(2 * math.pi * oscillation_freq * elapsed_time)
        angular_vel.y = 0.015 * math.sin(2 * math.pi * oscillation_freq * elapsed_time)
        angular_vel.z = 0.005 * math.cos(2 * math.pi * 0.2 * elapsed_time)

        orientation = self.quaternion_from_euler(roll, pitch, yaw)

        return linear_accel, angular_vel, orientation

    def update_drift(self):
        """Update drift values based on elapsed time"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time

        # Update drift with random walk characteristics
        self.current_accel_drift[0] += np.random.normal(0, self.accel_drift_rate[0])
        self.current_accel_drift[1] += np.random.normal(0, self.accel_drift_rate[1])
        self.current_accel_drift[2] += np.random.normal(0, self.accel_drift_rate[2])

        self.current_gyro_drift[0] += np.random.normal(0, self.gyro_drift_rate[0])
        self.current_gyro_drift[1] += np.random.normal(0, self.gyro_drift_rate[1])
        self.current_gyro_drift[2] += np.random.normal(0, self.gyro_drift_rate[2])

    def publish_imu(self):
        """Publish simulated IMU data with noise and drift"""
        # Update drift
        self.update_drift()

        # Simulate humanoid motion
        linear_acceleration, angular_velocity, orientation = self.simulate_humanoid_motion()

        # Add noise to readings
        noisy_accel, noisy_gyro, noisy_orientation = self.add_noise(
            linear_acceleration, angular_velocity, orientation
        )

        # Create IMU message
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        imu_msg.linear_acceleration = noisy_accel
        imu_msg.angular_velocity = noisy_gyro
        imu_msg.orientation = noisy_orientation

        # Set covariance matrices (diagonal values only for simplicity)
        # These represent the uncertainty in the measurements
        imu_msg.linear_acceleration_covariance = [
            self.linear_acceleration_noise**2, 0, 0,
            0, self.linear_acceleration_noise**2, 0,
            0, 0, self.linear_acceleration_noise**2
        ]

        imu_msg.angular_velocity_covariance = [
            self.angular_velocity_noise**2, 0, 0,
            0, self.angular_velocity_noise**2, 0,
            0, 0, self.angular_velocity_noise**2
        ]

        imu_msg.orientation_covariance = [
            self.orientation_noise**2, 0, 0,
            0, self.orientation_noise**2, 0,
            0, 0, self.orientation_noise**2
        ]

        # Publish the message
        self.imu_pub.publish(imu_msg)


def main(args=None):
    rclpy.init(args=args)
    imu_sim = ImuSimulator()

    try:
        rclpy.spin(imu_sim)
    except KeyboardInterrupt:
        imu_sim.get_logger().info('Shutting down IMU simulator')
    finally:
        imu_sim.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()