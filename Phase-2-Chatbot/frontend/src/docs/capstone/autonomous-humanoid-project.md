---
sidebar_position: 50
---

# Capstone Project: Autonomous Humanoid Robot System

## Learning Objectives
By the end of this capstone project, students will be able to:
- Integrate all components learned throughout the course into a complete humanoid robot system
- Implement a vision-language-action pipeline for natural human-robot interaction
- Deploy a complete AI-robot brain on a simulated humanoid platform
- Demonstrate autonomous navigation and manipulation in a domestic environment
- Evaluate and optimize system performance across all integrated components

## Project Overview

### The Challenge
Students will create an "Autonomous Humanoid" system that can:
1. Interpret natural language commands from users
2. Navigate safely through an apartment environment
3. Detect, identify, and manipulate household objects
4. Execute complex tasks requiring multiple steps and decision-making
5. Provide feedback and communicate with humans during task execution

### System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    Autonomous Humanoid System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Voice      │    │  Vision     │    │  Language   │         │
│  │  Input      │───▶│  Processing │───▶│  Processing │         │
│  │             │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Multi-Modal Fusion                      │   │
│  │  ┌─────────────────────────────────────────────────┐  │   │
│  │  │  Intent Understanding & Task Decomposition    │  │   │
│  │  └─────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Action Planning                      │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │   │
│  │  │ Navigation  │ │ Manipulation│ │ Human-Robot     │  │   │
│  │  │ Planning    │ │ Planning    │ │ Interaction     │  │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │         │         │         │                       │
│         ▼         ▼         ▼         ▼                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │Navigation│ │Arm/Hand  │ │Speech    │ │Behavior  │         │
│  │Control   │ │Control   │ │Synthesis │ │Manager   │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│         │         │         │         │                       │
│         └─────────┼─────────┼─────────┘                       │
│                     │         │                               │
│                     ▼         ▼                               │
│                ┌─────────────────────┐                        │
│                │   Robot Control     │                        │
│                │   (ROS 2 Nodes)     │                        │
│                └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Requirements
- ROS 2 Humble with all required packages
- NVIDIA Isaac Sim for simulation
- RealSense D435i camera (simulated)
- Unitree Go2 or similar humanoid robot model
- Gazebo for physics simulation
- OpenAI Whisper for voice processing
- LLM for task decomposition
- Custom controllers for bipedal locomotion

## Implementation

### Prerequisites
- Complete all previous modules (ROS 2, Digital Twin, AI-Brain, VLA)
- Ubuntu 22.04 with ROS 2 Humble
- NVIDIA GPU with RTX capability
- Isaac Sim installed and configured
- Basic understanding of all system components

### Phase 1: System Integration Framework

#### 1. Project Structure Setup
```bash
# Create capstone project structure
mkdir -p ~/ros2_ws/src/autonomous_humanoid
cd ~/ros2_ws/src/autonomous_humanoid

# Create package structure
mkdir -p autonomous_humanoid_bringup
mkdir -p autonomous_humanoid_perception
mkdir -p autonomous_humanoid_control
mkdir -p autonomous_humanoid_behavior
mkdir -p autonomous_humanoid_vla

# Create main launch directory
mkdir -p autonomous_humanoid_bringup/launch
mkdir -p autonomous_humanoid_bringup/config
```

#### 2. Main Launch File
Create `autonomous_humanoid_bringup/launch/autonomous_humanoid.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_model = LaunchConfiguration('robot_model', default='unitree_go2')
    apartment_world = LaunchConfiguration('apartment_world', default='apartment_world')

    # Paths
    pkg_share = FindPackageShare('autonomous_humanoid_bringup').find('autonomous_humanoid_bringup')
    rviz_config_path = PathJoinSubstitution([pkg_share, 'rviz', 'autonomous_humanoid.rviz'])

    # Include robot bringup
    robot_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('unitree_ros2'),
                'launch',
                'go2.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Include Gazebo simulation
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('apartment_worlds'),
                'worlds',
                [apartment_world, '.world']
            ]),
            'verbose': 'true'
        }.items()
    )

    # Perception pipeline
    perception_node = Node(
        package='autonomous_humanoid_perception',
        executable='perception_pipeline',
        name='perception_pipeline',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'camera_topic': '/camera/rgb/image_raw'},
            {'depth_topic': '/camera/depth/image_raw'},
            {'pointcloud_topic': '/camera/depth/points'}
        ],
        output='screen'
    )

    # Voice processing node
    voice_node = Node(
        package='autonomous_humanoid_vla',
        executable='voice_processor',
        name='voice_processor',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'audio_input_topic': '/microphone/audio_raw'},
            {'whisper_model_size': 'small'}
        ],
        output='screen'
    )

    # LLM task decomposition node
    llm_node = Node(
        package='autonomous_humanoid_vla',
        executable='llm_task_decomposer',
        name='llm_task_decomposer',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'llm_model': 'gpt-4'},
            {'max_tokens': 1000}
        ],
        output='screen'
    )

    # Behavior manager
    behavior_node = Node(
        package='autonomous_humanoid_behavior',
        executable='behavior_manager',
        name='behavior_manager',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'action_servers': [
                'navigation_controller',
                'manipulation_controller',
                'speech_controller'
            ]}
        ],
        output='screen'
    )

    # Navigation stack
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Timer actions for proper startup order
    delayed_perception = TimerAction(
        period=5.0,
        actions=[perception_node]
    )

    delayed_voice = TimerAction(
        period=8.0,
        actions=[voice_node]
    )

    delayed_llm = TimerAction(
        period=10.0,
        actions=[llm_node]
    )

    delayed_behavior = TimerAction(
        period=12.0,
        actions=[behavior_node]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'robot_model',
            default_value='unitree_go2',
            description='Robot model to use'
        ),
        DeclareLaunchArgument(
            'apartment_world',
            default_value='apartment_world',
            description='Apartment world to load'
        ),

        # Launch simulation and robot
        gazebo_launch,
        robot_bringup_launch,

        # Launch system components with delays
        delayed_perception,
        delayed_voice,
        delayed_llm,
        delayed_behavior,
        navigation_launch,
        rviz_node
    ])
```

#### 3. Behavior Manager Implementation
Create `autonomous_humanoid_behavior/src/behavior_manager.cpp`:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/move_group_action.hpp>
#include <action_msgs/msg/goal_status_array.hpp>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/xml_parsing.h>

class BehaviorManager : public rclcpp::Node
{
public:
    BehaviorManager() : Node("behavior_manager")
    {
        // Declare parameters
        this->declare_parameter<std::vector<std::string>>("action_servers",
            std::vector<std::string>{"navigation", "manipulation", "speech"});

        // Publishers
        command_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/behavior/command", 10);

        status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/behavior/status", 10);

        // Subscribers
        voice_command_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/voice_commands", 10,
            std::bind(&BehaviorManager::voiceCommandCallback, this, std::placeholders::_1));

        task_decomposition_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/task_decomposition", 10,
            std::bind(&BehaviorManager::taskDecompositionCallback, this, std::placeholders::_1));

        // Create behavior tree
        createBehaviorTree();

        RCLCPP_INFO(this->get_logger(), "Behavior manager initialized");
    }

private:
    void voiceCommandCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received voice command: %s", msg->data.c_str());

        // Process the command through the behavior tree
        processCommand(msg->data);
    }

    void taskDecompositionCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received task decomposition: %s", msg->data.c_str());

        // Parse task decomposition and execute
        executeTaskSequence(msg->data);
    }

    void processCommand(const std::string& command)
    {
        // Update behavior tree blackboard with new command
        bt_->blackboard()->set("current_command", command);

        // Run the behavior tree
        BT::NodeStatus status = bt_->tickOnce();

        std::string status_str = (status == BT::NodeStatus::SUCCESS) ? "SUCCESS" :
                                (status == BT::NodeStatus::FAILURE) ? "FAILURE" : "RUNNING";

        RCLCPP_INFO(this->get_logger(), "Behavior tree execution status: %s", status_str.c_str());
    }

    void executeTaskSequence(const std::string& task_sequence_json)
    {
        // Parse JSON and execute sequence of tasks
        // This would involve calling navigation, manipulation, etc.

        // For now, just publish status
        auto status_msg = std_msgs::msg::String();
        status_msg.data = "Executing task sequence: " + task_sequence_json;
        status_pub_->publish(status_msg);
    }

    void createBehaviorTree()
    {
        // Create behavior tree factory
        factory_ = std::make_shared<BT::BehaviorTreeFactory>();

        // Register custom nodes
        registerCustomNodes();

        // Create XML for behavior tree
        std::string xml_string = R"(
        <root BTCPP_format="4">
          <BehaviorTree>
            <Sequence name="MainSequence">
              <WaitForCommand/>
              <ParseTaskDecomposition/>
              <ExecuteTaskSequence/>
              <ReportCompletion/>
            </Sequence>
          </BehaviorTree>
        </root>
        )";

        // Create behavior tree
        bt_ = factory_->createTreeFromText(xml_string);

        RCLCPP_INFO(this->get_logger(), "Behavior tree created");
    }

    void registerCustomNodes()
    {
        // Register custom behavior tree nodes
        factory_->registerNodeType<WaitForCommandNode>("WaitForCommand");
        factory_->registerNodeType<ParseTaskDecompositionNode>("ParseTaskDecomposition");
        factory_->registerNodeType<ExecuteTaskSequenceNode>("ExecuteTaskSequence");
        factory_->registerNodeType<ReportCompletionNode>("ReportCompletion");
    }

    // Custom BT nodes
    class WaitForCommandNode : public BT::SyncActionNode
    {
    public:
        WaitForCommandNode(const std::string& name, const BT::NodeConfiguration& config) :
            BT::SyncActionNode(name, config) {}

        BT::NodeStatus tick() override
        {
            // Wait for command from blackboard
            std::string command;
            if (getInput("current_command", command) && !command.empty())
            {
                return BT::NodeStatus::SUCCESS;
            }
            return BT::NodeStatus::FAILURE;
        }

        static BT::PortsList providedPorts() { return {}; }
    };

    // Publishers and subscribers
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr voice_command_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr task_decomposition_sub_;

    // Behavior tree components
    std::shared_ptr<BT::BehaviorTreeFactory> factory_;
    std::unique_ptr<BT::Tree> bt_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BehaviorManager>());
    rclcpp::shutdown();
    return 0;
}
```

### Phase 2: Vision-Language-Action Integration

#### 1. VLA Pipeline Node
Create `autonomous_humanoid_vla/src/vla_pipeline.cpp`:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <openai/openai.hpp>
#include <nlohmann/json.hpp>

class VLAPipeline : public rclcpp::Node
{
public:
    VLAPipeline() : Node("vla_pipeline")
    {
        // Declare parameters
        this->declare_parameter<std::string>("openai_api_key", "");
        this->declare_parameter<std::string>("llm_model", "gpt-4-vision-preview");
        this->declare_parameter<double>("confidence_threshold", 0.7);

        // Get parameters
        api_key_ = this->get_parameter("openai_api_key").as_string();
        llm_model_ = this->get_parameter("llm_model").as_string();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();

        // Publishers
        task_decomposition_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/task_decomposition", 10);
        object_detection_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/object_detections", 10);

        // Subscribers
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/rgb/image_raw", 10,
            std::bind(&VLAPipeline::imageCallback, this, std::placeholders::_1));

        voice_command_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/voice_commands", 10,
            std::bind(&VLAPipeline::voiceCommandCallback, this, std::placeholders::_1));

        // Initialize OpenAI client if API key is provided
        if (!api_key_.empty())
        {
            openai::start();
            openai_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "OpenAI client initialized");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No OpenAI API key provided, VLA pipeline will be limited");
        }

        RCLCPP_INFO(this->get_logger(), "VLA pipeline initialized");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Store latest image for processing with voice commands
        latest_image_ = cv_ptr->image.clone();
        latest_image_timestamp_ = msg->header.stamp;
    }

    void voiceCommandCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        std::string command = msg->data;
        RCLCPP_INFO(this->get_logger(), "Processing voice command: %s", command.c_str());

        if (latest_image_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No image available for VLA processing");
            return;
        }

        // Perform vision-language-action processing
        processVLATask(command);
    }

    void processVLATask(const std::string& command)
    {
        if (openai_initialized_)
        {
            // Create message for OpenAI API
            auto chat_completion = openai::chat().create(
                {
                    {"model", llm_model_},
                    {"messages", {
                        {{"role", "system"}, {"content", system_prompt_}},
                        {{"role", "user"}, {"content", formatUserMessage(command)}}
                    }}
                });

            std::string response = chat_completion["choices"][0]["message"]["content"];

            // Publish task decomposition
            auto task_msg = std_msgs::msg::String();
            task_msg.data = response;
            task_decomposition_pub_->publish(task_msg);

            RCLCPP_INFO(this->get_logger(), "Task decomposition published");
        }
        else
        {
            // Fallback: create a simple task decomposition
            createFallbackTask(command);
        }
    }

    std::string formatUserMessage(const std::string& command)
    {
        // Convert OpenCV image to base64 for API
        std::vector<uchar> buffer;
        cv::imencode(".jpg", latest_image_, buffer);
        std::string image_base64 = cv::imencode(buffer);

        // Format message for API
        nlohmann::json message;
        message["type"] = "image_url";
        message["image_url"]["url"] = "data:image/jpeg;base64," + image_base64;

        nlohmann::json text_message;
        text_message["type"] = "text";
        text_message["text"] = "Command: " + command +
                              "\n\nDecompose this command into executable robot actions. " +
                              "Consider the visual scene and provide a sequence of actions " +
                              "the robot should perform.";

        nlohmann::json content_array = nlohmann::json::array();
        content_array.push_back(text_message);
        content_array.push_back(message);

        return content_array.dump();
    }

    void createFallbackTask(const std::string& command)
    {
        // Simple fallback task creation when OpenAI is not available
        std::string fallback_task = R"({
            "reasoning": "Simple fallback task decomposition",
            "action_sequence": [
                {
                    "step": 1,
                    "action": "navigate_to_position",
                    "parameters": {"location": "kitchen"},
                    "reason": "Go to kitchen based on command"
                },
                {
                    "step": 2,
                    "action": "object_detection",
                    "parameters": {"object_type": "unknown"},
                    "reason": "Look for relevant objects"
                },
                {
                    "step": 3,
                    "action": "report_status",
                    "parameters": {"message": "Completed basic task based on: " + command},
                    "reason": "Report completion"
                }
            ],
            "estimated_completion_time": 60,
            "potential_challenges": ["Limited understanding without vision-language model"]
        })";

        auto task_msg = std_msgs::msg::String();
        task_msg.data = fallback_task;
        task_decomposition_pub_->publish(task_msg);
    }

    // Publishers and subscribers
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr task_decomposition_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr object_detection_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr voice_command_sub_;

    // OpenAI integration
    std::string api_key_;
    std::string llm_model_;
    double confidence_threshold_;
    bool openai_initialized_ = false;

    // Latest image data
    cv::Mat latest_image_;
    builtin_interfaces::msg::Time latest_image_timestamp_;

    // System prompt for task decomposition
    std::string system_prompt_ =
        "You are an AI assistant for a humanoid robot. Your role is to decompose natural language commands "
        "into executable action sequences for the robot. The robot operates in an apartment environment "
        "and can perform navigation, manipulation, and interaction tasks. Consider the visual input "
        "when interpreting commands and provide a detailed sequence of actions with reasoning.";
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VLAPipeline>());
    rclcpp::shutdown();
    return 0;
}
```

### Phase 3: Control Integration

#### 1. Main Control Node
Create `autonomous_humanoid_control/src/main_controller.cpp`:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nlohmann/json.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class MainController : public rclcpp::Node
{
public:
    MainController() : Node("main_controller"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        // Publishers
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_commands", 10);
        status_pub_ = this->create_publisher<std_msgs::msg::String>("/robot_status", 10);

        // Subscribers
        task_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/task_decomposition", 10,
            std::bind(&MainController::taskCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&MainController::odometryCallback, this, std::placeholders::_1));

        // Timer for control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  // 20Hz control loop
            std::bind(&MainController::controlLoop, this));

        // Initialize state
        current_task_step_ = 0;
        is_executing_task_ = false;

        RCLCPP_INFO(this->get_logger(), "Main controller initialized");
    }

private:
    void taskCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        try {
            // Parse task decomposition JSON
            auto task_json = nlohmann::json::parse(msg->data);

            if (task_json.contains("action_sequence"))
            {
                current_task_sequence_ = task_json["action_sequence"];
                current_task_step_ = 0;
                is_executing_task_ = true;

                RCLCPP_INFO(this->get_logger(), "New task sequence received with %zu steps",
                           current_task_sequence_.size());
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error parsing task: %s", e.what());
        }
    }

    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Store current robot pose
        current_pose_ = msg->pose.pose;
        current_twist_ = msg->twist.twist;
    }

    void controlLoop()
    {
        if (!is_executing_task_ || current_task_sequence_.empty())
        {
            return;
        }

        if (current_task_step_ >= current_task_sequence_.size())
        {
            // Task completed
            finishTask();
            return;
        }

        // Execute current task step
        auto& current_action = current_task_sequence_[current_task_step_];

        if (executeAction(current_action))
        {
            // Action completed, move to next step
            current_task_step_++;

            if (current_task_step_ >= current_task_sequence_.size())
            {
                finishTask();
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Moving to task step %d", current_task_step_);
            }
        }
    }

    bool executeAction(const nlohmann::json& action)
    {
        std::string action_type = action.at("action");

        if (action_type == "navigate_to_position")
        {
            return executeNavigationAction(action);
        }
        else if (action_type == "object_detection")
        {
            return executeObjectDetectionAction(action);
        }
        else if (action_type == "object_grasping")
        {
            return executeGraspingAction(action);
        }
        else if (action_type == "object_placement")
        {
            return executePlacementAction(action);
        }
        else if (action_type == "speech_synthesis")
        {
            return executeSpeechAction(action);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Unknown action type: %s", action_type.c_str());
            return true; // Skip unknown actions
        }
    }

    bool executeNavigationAction(const nlohmann::json& action)
    {
        if (action.contains("parameters") && action["parameters"].contains("location"))
        {
            std::string location = action["parameters"]["location"];

            // In a real implementation, this would call navigation stack
            // For now, we'll just move in a general direction

            geometry_msgs::msg::Twist cmd_vel;
            cmd_vel.linear.x = 0.2;  // Move forward
            cmd_vel.angular.z = 0.0;

            cmd_vel_pub_->publish(cmd_vel);

            // Check if we're close enough to destination
            // This would involve checking navigation feedback in a real implementation
            static int nav_counter = 0;
            nav_counter++;

            if (nav_counter > 40)  // Simulate reaching destination after some time
            {
                nav_counter = 0;
                RCLCPP_INFO(this->get_logger(), "Navigation to %s completed", location.c_str());
                return true;
            }
        }
        return false;
    }

    bool executeObjectDetectionAction(const nlohmann::json& action)
    {
        // In a real implementation, this would call object detection
        // For now, we'll simulate completion

        std::string object_type = action["parameters"]["object_type"];
        RCLCPP_INFO(this->get_logger(), "Object detection for %s completed", object_type.c_str());

        return true;
    }

    bool executeGraspingAction(const nlohmann::json& action)
    {
        // In a real implementation, this would call manipulation stack
        std::string object = action["parameters"]["object"];
        RCLCPP_INFO(this->get_logger(), "Grasping object %s completed", object.c_str());

        return true;
    }

    bool executePlacementAction(const nlohmann::json& action)
    {
        // In a real implementation, this would call manipulation stack
        std::string location = action["parameters"]["location"];
        RCLCPP_INFO(this->get_logger(), "Object placement at %s completed", location.c_str());

        return true;
    }

    bool executeSpeechAction(const nlohmann::json& action)
    {
        if (action.contains("parameters") && action["parameters"].contains("text"))
        {
            std::string text = action["parameters"]["text"];
            RCLCPP_INFO(this->get_logger(), "Speaking: %s", text.c_str());

            // Publish status
            auto status_msg = std_msgs::msg::String();
            status_msg.data = "Speaking: " + text;
            status_pub_->publish(status_msg);
        }

        return true;
    }

    void finishTask()
    {
        is_executing_task_ = false;
        current_task_step_ = 0;
        current_task_sequence_.clear();

        // Stop robot
        geometry_msgs::msg::Twist stop_cmd;
        stop_cmd.linear.x = 0.0;
        stop_cmd.angular.z = 0.0;
        cmd_vel_pub_->publish(stop_cmd);

        RCLCPP_INFO(this->get_logger(), "Task completed successfully");

        // Publish completion status
        auto status_msg = std_msgs::msg::String();
        status_msg.data = "Task completed";
        status_pub_->publish(status_msg);
    }

    // Publishers and subscribers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr task_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // Control timer
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Robot state
    geometry_msgs::msg::Pose current_pose_;
    geometry_msgs::msg::Twist current_twist_;

    // Task execution state
    std::vector<nlohmann::json> current_task_sequence_;
    size_t current_task_step_;
    bool is_executing_task_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MainController>());
    rclcpp::shutdown();
    return 0;
}
```

### Phase 4: Testing and Evaluation

#### 1. System Integration Test
Create `test/integration_test.py`:

```python
#!/usr/bin/env python3
"""
Integration test for the autonomous humanoid system.
"""

import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, JointState
import time
import json


class AutonomousHumanoidTester(Node):
    """
    Test node for the autonomous humanoid system.
    """

    def __init__(self):
        super().__init__('autonomous_humanoid_tester')

        # Publishers
        self.voice_command_pub = self.create_publisher(
            String, '/voice_commands', 10
        )
        self.test_status_pub = self.create_publisher(
            String, '/test_status', 10
        )

        # Subscribers
        self.status_sub = self.create_subscription(
            String, '/robot_status', self.status_callback, 10
        )
        self.task_sub = self.create_subscription(
            String, '/task_decomposition', self.task_callback, 10
        )

        # Test state
        self.test_results = {}
        self.current_test = None
        self.test_start_time = None

        self.get_logger().info('Autonomous humanoid tester initialized')

    def status_callback(self, msg):
        """Process robot status updates."""
        if self.current_test:
            if "completed" in msg.data.lower() or "finished" in msg.data.lower():
                self.record_test_result(self.current_test, True, time.time() - self.test_start_time)
                self.current_test = None

    def task_callback(self, msg):
        """Process task decomposition updates."""
        try:
            task_data = json.loads(msg.data)
            if 'action_sequence' in task_data:
                self.get_logger().info(f'Task received with {len(task_data["action_sequence"])} steps')
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse task decomposition')

    def run_integration_tests(self):
        """Run comprehensive integration tests."""
        tests = [
            self.test_simple_navigation,
            self.test_object_detection,
            self.test_vla_integration,
            self.test_task_completion
        ]

        for test_func in tests:
            self.get_logger().info(f'Running test: {test_func.__name__}')
            test_func()
            time.sleep(2)  # Wait between tests

        self.print_test_summary()

    def test_simple_navigation(self):
        """Test simple navigation command."""
        self.current_test = 'simple_navigation'
        self.test_start_time = time.time()

        command_msg = String()
        command_msg.data = "Go to the kitchen"
        self.voice_command_pub.publish(command_msg)

        # Wait for completion or timeout
        timeout = time.time() + 30  # 30 second timeout
        while self.current_test and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_test:
            # Test timed out
            self.record_test_result(self.current_test, False, 30.0)
            self.current_test = None

    def test_object_detection(self):
        """Test object detection capability."""
        self.current_test = 'object_detection'
        self.test_start_time = time.time()

        command_msg = String()
        command_msg.data = "Find the red cup in the living room"
        self.voice_command_pub.publish(command_msg)

        # Wait for completion or timeout
        timeout = time.time() + 45  # 45 second timeout
        while self.current_test and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_test:
            # Test timed out
            self.record_test_result(self.current_test, False, 45.0)
            self.current_test = None

    def test_vla_integration(self):
        """Test vision-language-action integration."""
        self.current_test = 'vla_integration'
        self.test_start_time = time.time()

        command_msg = String()
        command_msg.data = "Bring me a glass of water from the kitchen and place it on the coffee table"
        self.voice_command_pub.publish(command_msg)

        # Wait for completion or timeout
        timeout = time.time() + 90  # 90 second timeout
        while self.current_test and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_test:
            # Test timed out
            self.record_test_result(self.current_test, False, 90.0)
            self.current_test = None

    def test_task_completion(self):
        """Test complex multi-step task completion."""
        self.current_test = 'task_completion'
        self.test_start_time = time.time()

        command_msg = String()
        command_msg.data = "Go to the bedroom, pick up the book from the nightstand, bring it to me in the living room"
        self.voice_command_pub.publish(command_msg)

        # Wait for completion or timeout
        timeout = time.time() + 120  # 120 second timeout
        while self.current_test and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_test:
            # Test timed out
            self.record_test_result(self.current_test, False, 120.0)
            self.current_test = None

    def record_test_result(self, test_name, success, duration):
        """Record test result."""
        self.test_results[test_name] = {
            'success': success,
            'duration': duration,
            'timestamp': time.time()
        }

        status_msg = String()
        status_msg.data = f'Test {test_name}: {"PASSED" if success else "FAILED"} in {duration:.2f}s'
        self.test_status_pub.publish(status_msg)

        self.get_logger().info(f'{status_msg.data}')

    def print_test_summary(self):
        """Print test results summary."""
        passed = sum(1 for result in self.test_results.values() if result['success'])
        total = len(self.test_results)

        summary = f'\n=== Integration Test Summary ===\n'
        summary += f'Tests Passed: {passed}/{total}\n'
        summary += f'Success Rate: {passed/total*100:.1f}%\n'
        summary += f'===============================\n'

        for test_name, result in self.test_results.items():
            status = "PASSED" if result['success'] else "FAILED"
            summary += f'{test_name}: {status} ({result["duration"]:.2f}s)\n'

        self.get_logger().info(summary)


def main(args=None):
    rclpy.init(args=args)

    tester = AutonomousHumanoidTester()

    # Run tests
    tester.run_integration_tests()

    # Keep node alive briefly to publish final results
    time.sleep(5)

    tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### 2. Performance Evaluation Script
Create `scripts/performance_evaluator.py`:

```python
#!/usr/bin/env python3
"""
Performance evaluation script for the autonomous humanoid system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
import time
import statistics
from collections import deque
import matplotlib.pyplot as plt
import numpy as np


class PerformanceEvaluator(Node):
    """
    Evaluate performance of the autonomous humanoid system.
    """

    def __init__(self):
        super().__init__('performance_evaluator')

        # Parameters
        self.declare_parameter('evaluation_duration', 300)  # 5 minutes
        self.declare_parameter('metrics_output_path', '/tmp/performance_metrics.json')

        self.evaluation_duration = self.get_parameter('evaluation_duration').value
        self.metrics_output_path = self.get_parameter('metrics_output_path').value

        # Publishers
        self.metrics_pub = self.create_publisher(Float32, '/performance/metrics', 10)
        self.status_pub = self.create_publisher(String, '/performance/status', 10)

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, '/voice_commands', self.voice_callback, 10
        )
        self.status_sub = self.create_subscription(
            String, '/robot_status', self.status_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # Metrics tracking
        self.command_response_times = deque(maxlen=1000)
        self.image_processing_times = deque(maxlen=1000)
        self.navigation_success_rates = deque(maxlen=100)
        self.task_completion_times = deque(maxlen=100)
        self.system_uptime = 0.0

        # Evaluation state
        self.evaluation_start_time = time.time()
        self.current_task_start_time = None
        self.task_count = 0

        # Timer for periodic evaluation
        self.eval_timer = self.create_timer(1.0, self.periodic_evaluation)

        self.get_logger().info('Performance evaluator initialized')

    def voice_callback(self, msg):
        """Track command arrival time."""
        self.command_arrival_time = time.time()

    def status_callback(self, msg):
        """Process status updates and calculate response time."""
        if hasattr(self, 'command_arrival_time'):
            response_time = time.time() - self.command_arrival_time
            self.command_response_times.append(response_time)

            # Record task completion if indicated
            if 'completed' in msg.data.lower():
                if self.current_task_start_time:
                    completion_time = time.time() - self.current_task_start_time
                    self.task_completion_times.append(completion_time)
                    self.current_task_start_time = None
                    self.task_count += 1

        # Track navigation success/failure
        if 'navigation' in msg.data.lower():
            if 'success' in msg.data.lower() or 'completed' in msg.data.lower():
                self.navigation_success_rates.append(1.0)
            elif 'failed' in msg.data.lower() or 'aborted' in msg.data.lower():
                self.navigation_success_rates.append(0.0)

    def image_callback(self, msg):
        """Track image processing."""
        # This would track actual image processing time in a real system
        self.image_processing_times.append(0.033)  # Assuming 30 FPS

    def joint_callback(self, msg):
        """Track joint state updates."""
        # This could track control loop performance
        pass

    def periodic_evaluation(self):
        """Perform periodic evaluation and publish metrics."""
        current_time = time.time()
        self.system_uptime = current_time - self.evaluation_start_time

        # Calculate metrics
        metrics = self.calculate_current_metrics()

        # Publish overall performance score
        perf_msg = Float32()
        perf_msg.data = metrics['overall_score']
        self.metrics_pub.publish(perf_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Performance Score: {metrics['overall_score']:.2f}, " \
                         f"Uptime: {self.system_uptime:.1f}s, " \
                         f"Tasks: {self.task_count}"
        self.status_pub.publish(status_msg)

        # Log metrics periodically
        if int(self.system_uptime) % 30 == 0:  # Log every 30 seconds
            self.get_logger().info(
                f"Performance Metrics:\n"
                f"  Response Time: {metrics['avg_response_time']:.3f}s\n"
                f"  Success Rate: {metrics['success_rate']:.2f}\n"
                f"  Task Time: {metrics['avg_task_time']:.2f}s\n"
                f"  Overall Score: {metrics['overall_score']:.2f}"
            )

    def calculate_current_metrics(self):
        """Calculate current performance metrics."""
        metrics = {}

        # Response time metrics
        if self.command_response_times:
            metrics['avg_response_time'] = statistics.mean(self.command_response_times)
            metrics['std_response_time'] = statistics.stdev(self.command_response_times) if len(self.command_response_times) > 1 else 0
            metrics['min_response_time'] = min(self.command_response_times)
            metrics['max_response_time'] = max(self.command_response_times)
        else:
            metrics['avg_response_time'] = 0.0
            metrics['std_response_time'] = 0.0
            metrics['min_response_time'] = 0.0
            metrics['max_response_time'] = 0.0

        # Success rate metrics
        if self.navigation_success_rates:
            metrics['success_rate'] = sum(self.navigation_success_rates) / len(self.navigation_success_rates)
        else:
            metrics['success_rate'] = 0.0

        # Task completion metrics
        if self.task_completion_times:
            metrics['avg_task_time'] = statistics.mean(self.task_completion_times)
            metrics['std_task_time'] = statistics.stdev(self.task_completion_times) if len(self.task_completion_times) > 1 else 0
        else:
            metrics['avg_task_time'] = 0.0
            metrics['std_task_time'] = 0.0

        # Overall performance score (weighted combination of key metrics)
        response_weight = 0.3
        success_weight = 0.4
        time_weight = 0.3

        # Normalize response time (lower is better, target < 2.0s)
        norm_response = max(0, min(1, (2.0 - metrics['avg_response_time']) / 2.0))

        # Success rate is already 0-1
        norm_success = metrics['success_rate']

        # Normalize task time (target < 60s for typical tasks)
        norm_time = max(0, min(1, (60.0 - metrics['avg_task_time']) / 60.0))

        metrics['overall_score'] = (
            response_weight * norm_response +
            success_weight * norm_success +
            time_weight * norm_time
        ) * 100  # Scale to 0-100

        return metrics

    def save_final_metrics(self):
        """Save final metrics to file."""
        import json

        final_metrics = self.calculate_current_metrics()
        final_metrics['total_uptime'] = self.system_uptime
        final_metrics['total_tasks_completed'] = self.task_count
        final_metrics['evaluation_duration'] = self.system_uptime

        with open(self.metrics_output_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)

        self.get_logger().info(f'Final metrics saved to {self.metrics_output_path}')


def main(args=None):
    rclpy.init(args=args)

    evaluator = PerformanceEvaluator()

    # Run evaluation for specified duration
    start_time = time.time()
    evaluation_duration = evaluator.evaluation_duration

    try:
        while time.time() - start_time < evaluation_duration:
            rclpy.spin_once(evaluator, timeout_sec=0.1)
    except KeyboardInterrupt:
        evaluator.get_logger().info('Performance evaluation interrupted by user')
    finally:
        # Save final metrics
        evaluator.save_final_metrics()
        evaluator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Phase 5: Final System Verification

#### 1. Complete System Test
Create `scripts/final_verification.py`:

```python
#!/usr/bin/env python3
"""
Final verification script for the autonomous humanoid system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist
import time
import json
import subprocess


class FinalVerification(Node):
    """
    Final verification for the complete autonomous humanoid system.
    """

    def __init__(self):
        super().__init__('final_verification')

        # Publishers
        self.command_pub = self.create_publisher(String, '/voice_commands', 10)
        self.verification_pub = self.create_publisher(Bool, '/verification/status', 10)
        self.score_pub = self.create_publisher(Float32, '/verification/score', 10)

        # Subscribers
        self.status_sub = self.create_subscription(
            String, '/robot_status', self.status_callback, 10
        )
        self.task_sub = self.create_subscription(
            String, '/task_decomposition', self.task_callback, 10
        )

        # Verification state
        self.verification_results = {}
        self.current_scenario = None
        self.scenario_start_time = None
        self.verification_score = 0.0

        self.get_logger().info('Final verification system initialized')

    def run_comprehensive_verification(self):
        """Run comprehensive verification of all system components."""

        # Define verification scenarios
        scenarios = [
            {
                "name": "basic_navigation",
                "command": "Go to the kitchen",
                "expected_actions": ["navigate_to_position"],
                "timeout": 60
            },
            {
                "name": "object_interaction",
                "command": "Find the red cup in the living room and tell me where it is",
                "expected_actions": ["object_detection", "speech_synthesis"],
                "timeout": 90
            },
            {
                "name": "complex_task",
                "command": "Bring me a glass of water from the kitchen and place it on the coffee table in the living room",
                "expected_actions": ["navigate_to_position", "object_detection", "object_grasping", "object_placement"],
                "timeout": 180
            },
            {
                "name": "multi_room_navigation",
                "command": "Go from the living room to the bedroom, then to the kitchen, and return to the living room",
                "expected_actions": ["navigate_to_position"] * 3,
                "timeout": 120
            }
        ]

        total_score = 0.0
        scenario_count = len(scenarios)

        for scenario in scenarios:
            self.get_logger().info(f'Running verification scenario: {scenario["name"]}')
            score = self.verify_scenario(scenario)
            self.verification_results[scenario["name"]] = score
            total_score += score

        # Calculate final score
        final_score = total_score / scenario_count if scenario_count > 0 else 0.0
        self.verification_score = final_score

        # Publish results
        self.publish_verification_results()

        return final_score

    def verify_scenario(self, scenario):
        """Verify a single scenario."""
        self.current_scenario = scenario
        self.scenario_start_time = time.time()

        # Publish command
        command_msg = String()
        command_msg.data = scenario["command"]
        self.command_pub.publish(command_msg)

        # Wait for completion or timeout
        timeout_time = time.time() + scenario["timeout"]
        actions_received = []

        while time.time() < timeout_time:
            rclpy.spin_once(self, timeout_sec=0.1)

            # Check if scenario is complete
            if self.is_scenario_complete(scenario, actions_received):
                elapsed_time = time.time() - self.scenario_start_time
                success = True
                break
        else:
            # Timeout occurred
            elapsed_time = scenario["timeout"]
            success = False

        # Calculate scenario score
        score = self.calculate_scenario_score(scenario, success, elapsed_time, actions_received)

        self.get_logger().info(
            f'Scenario {scenario["name"]}: {"PASSED" if success else "FAILED"} | '
            f'Score: {score:.2f} | Time: {elapsed_time:.2f}s'
        )

        return score

    def is_scenario_complete(self, scenario, actions_received):
        """Check if scenario is complete based on expected actions."""
        # For simplicity, we'll check if we've received status indicating completion
        # In a real system, this would be more sophisticated
        return len(actions_received) >= len(scenario["expected_actions"])

    def calculate_scenario_score(self, scenario, success, elapsed_time, actions_received):
        """Calculate score for a scenario."""
        score = 0.0

        if success:
            # Base score for success
            score += 70.0

            # Time bonus for completing faster
            time_bonus = max(0, (scenario["timeout"] - elapsed_time) / scenario["timeout"] * 20.0)
            score += time_bonus

            # Action completeness bonus
            expected_count = len(scenario["expected_actions"])
            received_count = len(actions_received)
            if received_count >= expected_count:
                score += 10.0
        else:
            # Partial credit for partial completion
            expected_count = len(scenario["expected_actions"])
            received_count = len(actions_received)
            if expected_count > 0:
                partial_credit = (received_count / expected_count) * 50.0
                score += partial_credit

        return min(100.0, score)  # Cap at 100

    def status_callback(self, msg):
        """Process status messages for verification."""
        if self.current_scenario:
            # Record actions as they're reported
            if "completed" in msg.data.lower():
                # Scenario step completed
                pass

    def task_callback(self, msg):
        """Process task decomposition for verification."""
        try:
            task_data = json.loads(msg.data)
            if 'action_sequence' in task_data:
                # Record that we received a task decomposition
                pass
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse task for verification')

    def publish_verification_results(self):
        """Publish verification results."""
        # Publish overall status
        status_msg = Bool()
        status_msg.data = self.verification_score >= 80.0  # Pass threshold
        self.verification_pub.publish(status_msg)

        # Publish score
        score_msg = Float32()
        score_msg.data = self.verification_score
        self.score_pub.publish(score_msg)

        # Print results
        self.print_verification_summary()

    def print_verification_summary(self):
        """Print verification results summary."""
        summary = f'\n=== Final Verification Results ===\n'
        summary += f'Overall Score: {self.verification_score:.2f}/100\n'
        summary += f'Status: {"PASSED" if self.verification_score >= 80.0 else "FAILED"}\n'
        summary += f'==================================\n'

        for scenario, score in self.verification_results.items():
            status = "PASS" if score >= 70.0 else "FAIL"
            summary += f'{scenario}: {status} ({score:.2f}/100)\n'

        self.get_logger().info(summary)

        # Save results to file
        import json
        results_path = '/tmp/final_verification_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'overall_score': self.verification_score,
                'status': 'PASS' if self.verification_score >= 80.0 else 'FAIL',
                'scenarios': self.verification_results,
                'timestamp': time.time()
            }, f, indent=2)

        self.get_logger().info(f'Verification results saved to {results_path}')


def main(args=None):
    rclpy.init(args=args)

    verifier = FinalVerification()

    try:
        final_score = verifier.run_comprehensive_verification()
        verifier.get_logger().info(f'Final verification score: {final_score:.2f}')
    except KeyboardInterrupt:
        verifier.get_logger().info('Verification interrupted by user')
    finally:
        verifier.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercises

1. Implement the complete autonomous humanoid system with all components
2. Test the system with various natural language commands in simulation
3. Evaluate system performance under different environmental conditions
4. Optimize the VLA pipeline for real-time performance
5. Implement error handling and recovery mechanisms
6. Test the system's ability to handle ambiguous or complex commands
7. Evaluate the system's performance in multi-room navigation scenarios
8. Create additional test scenarios to validate edge cases

## References

1. Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html
2. ROS 2 Navigation: https://navigation.ros.org/
3. Behavior Trees in Robotics: https://arxiv.org/abs/1709.00084
4. Vision-Language-Action Models: https://arxiv.org/abs/2206.04689

## Further Reading

- Advanced robotics simulation techniques with Isaac Sim
- Multi-modal learning for robotics applications
- Human-robot interaction design principles
- Real-time performance optimization for robotic systems
- Integration of large language models with robotic control systems