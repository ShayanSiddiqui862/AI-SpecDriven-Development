#!/usr/bin/env python3

"""
Example ROS 2 publisher node for the Physical AI & Humanoid Robotics textbook.

This node demonstrates basic ROS 2 publisher functionality using rclpy.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ExamplePublisher(Node):
    """
    A simple ROS 2 publisher node that publishes messages to a topic.
    """

    def __init__(self):
        """Initialize the publisher node."""
        super().__init__('example_publisher')
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        """Callback function that publishes messages."""
        msg = String()
        msg.data = f'Robot is operational: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    """Main function to run the example publisher node."""
    rclpy.init(args=args)

    example_publisher = ExamplePublisher()

    try:
        rclpy.spin(example_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        example_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()