---
sidebar_position: 12
---

# rclpy Python Agent Tutorials

## Learning Objectives
By the end of this module, students will be able to:
- Create ROS 2 nodes using rclpy
- Implement publishers and subscribers for topic-based communication
- Develop services and clients for request/response communication
- Create action servers and clients for goal-oriented tasks
- Use parameters for node configuration
- Organize nodes with launch files

## Theory

### rclpy Overview
rclpy is the Python client library for ROS 2, providing a Python API for ROS concepts. It allows Python developers to create ROS 2 nodes that can communicate with other nodes using topics, services, and actions.

### Key Concepts
- **Nodes**: Basic computational elements that execute ROS programs
- **Topics**: Named buses over which nodes exchange messages (publish/subscribe)
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous, goal-oriented communication with feedback
- **Parameters**: Configuration values that can be set at runtime

## Implementation

### Prerequisites
- ROS 2 Humble installed
- Basic Python programming knowledge
- Understanding of ROS 2 concepts

### Example 1: Simple Publisher/Subscriber

#### Publisher Node
Create `simple_publisher.py`:

```python
#!/usr/bin/env python3

"""
Simple publisher node demonstrating basic rclpy usage.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    """
    A simple publisher that sends messages to a topic.
    """

    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    simple_publisher = SimplePublisher()

    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        simple_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Subscriber Node
Create `simple_subscriber.py`:

```python
#!/usr/bin/env python3

"""
Simple subscriber node demonstrating basic rclpy usage.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    """
    A simple subscriber that receives messages from a topic.
    """

    def __init__(self):
        super().__init__('simple_subscriber')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)

    simple_subscriber = SimpleSubscriber()

    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        simple_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Running the Example
Terminal 1:
```bash
python3 simple_publisher.py
```

Terminal 2:
```bash
python3 simple_subscriber.py
```

### Example 2: Service/Client

#### Service Server
Create `add_two_ints_server.py`:

```python
#!/usr/bin/env python3

"""
Service server for adding two integers.
"""

import sys
import rclpy
from rclpy.node import Node

from example_interfaces.srv import AddTwoInts


class AddTwoIntsServer(Node):
    """
    Service server that adds two integers.
    """

    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)

    add_two_ints_server = AddTwoIntsServer()

    try:
        rclpy.spin(add_two_ints_server)
    except KeyboardInterrupt:
        pass
    finally:
        add_two_ints_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Service Client
Create `add_two_ints_client.py`:

```python
#!/usr/bin/env python3

"""
Service client for adding two integers.
"""

import sys
import rclpy
from rclpy.node import Node

from example_interfaces.srv import AddTwoInts


class AddTwoIntsClient(Node):
    """
    Service client that calls the add_two_ints service.
    """

    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main():
    rclpy.init()

    add_two_ints_client = AddTwoIntsClient()
    response = add_two_ints_client.send_request(int(sys.argv[1]), int(sys.argv[2]))

    if response is not None:
        add_two_ints_client.get_logger().info(
            f'Result of add_two_ints: {response.sum}')
    else:
        add_two_ints_client.get_logger().info('Service call failed')

    add_two_ints_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Example 3: Action Server/Client

#### Action Server
Create `fibonacci_action_server.py`:

```python
#!/usr/bin/env python3

"""
Fibonacci action server example.
"""

import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):
    """
    Fibonacci action server that generates a sequence of Fibonacci numbers.
    """

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result


def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    try:
        executor = MultiThreadedExecutor()
        rclpy.spin(fibonacci_action_server, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Action Client
Create `fibonacci_action_client.py`:

```python
#!/usr/bin/env python3

"""
Fibonacci action client example.
"""

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):
    """
    Fibonacci action client that requests a sequence of Fibonacci numbers.
    """

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci',
            callback_group=ReentrantCallbackGroup())

    def send_goal(self, order):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.get_logger().info(f'Sending goal request with order {order}...')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    action_client = FibonacciActionClient()

    # Spin in a separate thread
    executor = MultiThreadedExecutor()
    executor.add_node(action_client)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(1)  # Wait for the action server to be available

    action_client.send_goal(10)

    try:
        # Wait for the result
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


from threading import Thread


if __name__ == '__main__':
    main()
```

### Example 4: Parameters

Create `parameter_node.py`:

```python
#!/usr/bin/env python3

"""
Parameter example demonstrating parameter declaration and usage.
"""

import rclpy
from rclpy.node import Node


class ParameterNode(Node):
    """
    Node that demonstrates parameter usage.
    """

    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('use_sim_time', False)

        # Get parameter values
        robot_name = self.get_parameter('robot_name').value
        max_velocity = self.get_parameter('max_velocity').value
        use_sim_time = self.get_parameter('use_sim_time').value

        self.get_logger().info(f'Robot name: {robot_name}')
        self.get_logger().info(f'Max velocity: {max_velocity}')
        self.get_logger().info(f'Use sim time: {use_sim_time}')

        # Set up parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """
        Callback for parameter changes.
        """
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)

    parameter_node = ParameterNode()

    try:
        rclpy.spin(parameter_node)
    except KeyboardInterrupt:
        pass
    finally:
        parameter_node.destroy_node()
        rclpy.shutdown()


from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_parameters
from rcl_interfaces.msg import SetParametersResult


if __name__ == '__main__':
    main()
```

### Example 5: Launch Files

Create `robot_launch_example.py`:

```python
#!/usr/bin/env python3

"""
Launch file example demonstrating how to launch multiple nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'),

        # Launch the publisher node
        Node(
            package='robot_examples',
            executable='simple_publisher',
            name='talker',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        ),

        # Launch the subscriber node
        Node(
            package='robot_examples',
            executable='simple_subscriber',
            name='listener',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        )
    ])
```

### Example 6: Robot Controller Node

Create `robot_controller.py`:

```python
#!/usr/bin/env python3

"""
Robot controller example that demonstrates more complex ROS 2 usage.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math


class RobotController(Node):
    """
    A more complex robot controller node.
    """

    def __init__(self):
        super().__init__('robot_controller')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.position = [0.0, 0.0]
        self.orientation = 0.0
        self.laser_data = None
        self.target = [2.0, 2.0]  # Target position

        self.get_logger().info('Robot controller initialized')

    def odom_callback(self, msg):
        """Callback for odometry data."""
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y

        # Extract orientation from quaternion
        quat = msg.pose.pose.orientation
        self.orientation = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )

    def scan_callback(self, msg):
        """Callback for laser scan data."""
        self.laser_data = msg.ranges

    def control_loop(self):
        """Main control loop."""
        if self.laser_data is None:
            return

        # Simple obstacle avoidance and navigation
        cmd_vel = Twist()

        # Calculate distance to target
        dist_to_target = math.sqrt(
            (self.target[0] - self.position[0])**2 +
            (self.target[1] - self.position[1])**2
        )

        # If close to target, stop
        if dist_to_target < 0.5:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        else:
            # Navigate toward target with obstacle avoidance
            target_angle = math.atan2(
                self.target[1] - self.position[1],
                self.target[0] - self.position[0]
            )

            angle_diff = target_angle - self.orientation
            # Normalize angle
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Simple proportional controller
            cmd_vel.angular.z = 0.5 * angle_diff

            # Check for obstacles
            min_distance = min(self.laser_data) if self.laser_data else float('inf')

            if min_distance < 0.5:  # Obstacle too close
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.5  # Turn away from obstacle
            else:
                cmd_vel.linear.x = min(0.5, dist_to_target * 0.5)  # Move toward target

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)

    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercises

1. Implement all six examples and run them to understand the different communication patterns
2. Modify the publisher/subscriber example to publish sensor data instead of strings
3. Create a custom message type and use it in a publisher/subscriber pair
4. Implement a service that performs a more complex calculation (e.g., path planning)
5. Extend the robot controller to handle multiple targets in sequence
6. Create a launch file that starts multiple robot controllers in a simulated environment

## References

1. ROS 2 Python Client Library (rclpy) Documentation: https://docs.ros.org/en/humble/p/rclpy/
2. ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
3. ROS 2 Concepts: https://docs.ros.org/en/humble/Concepts.html

## Further Reading

- ROS 2 Quality of Service (QoS) settings
- ROS 2 security features
- Advanced rclpy patterns and best practices
- Integration with other Python libraries (NumPy, OpenCV, etc.)