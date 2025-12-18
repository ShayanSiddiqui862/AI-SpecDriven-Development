#!/usr/bin/env python3

"""
Example ROS 2 subscriber node for the Physical AI & Humanoid Robotics textbook.

This node demonstrates basic ROS 2 subscriber functionality using rclpy.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ExampleSubscriber(Node):
    """
    A simple ROS 2 subscriber node that subscribes to messages from a topic.
    """

    def __init__(self):
        """Initialize the subscriber node."""
        super().__init__('example_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        """Callback function that processes incoming messages."""
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Main function to run the example subscriber node."""
    rclpy.init(args=args)

    example_subscriber = ExampleSubscriber()

    try:
        rclpy.spin(example_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        example_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()