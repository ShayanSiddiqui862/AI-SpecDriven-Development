from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command([
        'xacro ',
        FindPackageShare('bipedal_controllers').find('bipedal_controllers'),
        '/urdf/humanoid.urdf.xacro'
    ])
    robot_description = {'robot_description': robot_description_content}

    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description]
    )

    # Controller Manager
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description,
                   os.path.join(FindPackageShare('bipedal_controllers'), 'config', 'config.yaml')],
        output='both'
    )

    # Joint State Broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )

    # Left Leg Controller
    left_leg_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_leg_controller', '--controller-manager', '/controller_manager'],
    )

    # Right Leg Controller
    right_leg_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_leg_controller', '--controller-manager', '/controller_manager'],
    )

    # Balance Controller
    balance_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['balance_controller', '--controller-manager', '/controller_manager'],
    )

    # Delay start of joint_state_broadcaster after robot_state_publisher
    delay_joint_state_broadcaster_after_robot_state_publisher = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_state_publisher_node,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    # Delay start of controllers after joint_state_broadcaster
    delay_left_leg_controller_after_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[left_leg_controller_spawner],
        )
    )

    delay_right_leg_controller_after_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[right_leg_controller_spawner],
        )
    )

    delay_balance_controller_after_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[balance_controller_spawner],
        )
    )

    # Return LaunchDescription
    return LaunchDescription([
        robot_state_publisher_node,
        controller_manager_node,
        delay_joint_state_broadcaster_after_robot_state_publisher,
        delay_left_leg_controller_after_joint_state_broadcaster,
        delay_right_leg_controller_after_joint_state_broadcaster,
        delay_balance_controller_after_joint_state_broadcaster,
    ])