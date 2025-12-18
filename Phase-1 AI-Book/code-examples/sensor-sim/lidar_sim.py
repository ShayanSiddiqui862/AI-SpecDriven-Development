#!/usr/bin/env python3

"""
LiDAR Sensor Simulation
Implements realistic LiDAR sensor with noise models for humanoid robotics
"""

import numpy as np
import math
import random
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import rclpy
from rclpy.node import Node


class LidarSimulator(Node):
    def __init__(self):
        super().__init__('lidar_simulator')

        # Publisher for simulated LiDAR data
        self.lidar_pub = self.create_publisher(LaserScan, '/scan', 10)

        # LiDAR parameters
        self.angle_min = -math.pi
        self.angle_max = math.pi
        self.angle_increment = 0.017453292519943295  # 1 degree
        self.scan_time = 0.1
        self.range_min = 0.1
        self.range_max = 30.0

        # Noise parameters
        self.range_noise_std = 0.02  # 2cm standard deviation
        self.intensity_noise_std = 10.0

        # Timer for publishing data
        self.timer = self.create_timer(0.1, self.publish_scan)

        self.get_logger().info('LiDAR Simulator initialized')

    def add_noise(self, ranges, intensities):
        """Add realistic noise to sensor readings"""
        noisy_ranges = []
        noisy_intensities = []

        for i, range_val in enumerate(ranges):
            if not (self.range_min <= range_val <= self.range_max):
                # Invalid range, keep as is
                noisy_ranges.append(range_val)
            else:
                # Add Gaussian noise to range
                noise = np.random.normal(0, self.range_noise_std)
                noisy_range = max(self.range_min, min(self.range_max, range_val + noise))
                noisy_ranges.append(noisy_range)

        for intensity in intensities:
            # Add noise to intensity
            noise = np.random.normal(0, self.intensity_noise_std)
            noisy_intensity = max(0, intensity + noise)
            noisy_intensities.append(noisy_intensity)

        return noisy_ranges, noisy_intensities

    def simulate_environment(self):
        """Simulate a simple environment for the LiDAR to scan"""
        num_readings = int((self.angle_max - self.angle_min) / self.angle_increment) + 1
        ranges = []
        intensities = []

        # Simulate some obstacles in the environment
        for i in range(num_readings):
            angle = self.angle_min + i * self.angle_increment

            # Simulate distance to nearest obstacle at this angle
            # This creates a simple environment with a few obstacles
            distance = self.range_max  # Default to max range (free space)

            # Add some obstacles (simulated walls, objects)
            if -0.5 < angle < 0.5:  # Front of robot
                # Wall in front
                distance = 5.0 + 0.5 * math.sin(10 * angle)  # Slightly curved wall
            elif 1.0 < abs(angle) < 1.5:  # Sides
                distance = 3.0 + random.uniform(-0.2, 0.2)  # Side obstacles

            ranges.append(distance)
            intensities.append(100.0)  # Default intensity

        return ranges, intensities

    def publish_scan(self):
        """Publish simulated LiDAR scan with noise"""
        # Simulate environment
        ranges, intensities = self.simulate_environment()

        # Add noise to readings
        noisy_ranges, noisy_intensities = self.add_noise(ranges, intensities)

        # Create LaserScan message
        scan_msg = LaserScan()
        scan_msg.header = Header()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'

        scan_msg.angle_min = self.angle_min
        scan_msg.angle_max = self.angle_max
        scan_msg.angle_increment = self.angle_increment
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = self.scan_time
        scan_msg.range_min = self.range_min
        scan_msg.range_max = self.range_max

        scan_msg.ranges = noisy_ranges
        scan_msg.intensities = noisy_intensities

        # Publish the message
        self.lidar_pub.publish(scan_msg)


def main(args=None):
    rclpy.init(args=args)
    lidar_sim = LidarSimulator()

    try:
        rclpy.spin(lidar_sim)
    except KeyboardInterrupt:
        lidar_sim.get_logger().info('Shutting down LiDAR simulator')
    finally:
        lidar_sim.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()