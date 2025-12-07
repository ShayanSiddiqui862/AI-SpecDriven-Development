// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1 - ROS 2 Humanoid Control',
      items: [
        'module1/index',
        'module1/ros2-installation',
        'module1/rclpy-agents',
        'module1/urdf-humanoids',
        'module1/bipedal-control',
        'module1/verification'
      ],
    },
    {
      type: 'category',
      label: 'Module 2 - Digital Twin Environment Setup',
      items: [
        'module2/index',
        'module2/gazebo-worlds',
        'module2/urdf-to-sdf',
        'module2/sensor-simulation',
        'module2/unity-hri',
        'module2/verification'
      ],
    },
    {
      type: 'category',
      label: 'Module 3 - AI-Robot Brain Development',
      items: [
        'module3/index',
        'module3/isaac-sim-deployment',
        'module3/synthetic-data',
        'module3/synthetic-data-generation',
        'module3/vslam-realsense',
        'module3/vslam-implementation',
        'module3/nav2-bipedal',
        'module3/nav2-bipedal-config',
        'module3/perception-navigation-validation'
      ],
    },
    {
      type: 'category',
      label: 'Module 4 - Vision-Language-Action System',
      items: [
        'module4/index',
        'module4/whisper-ros2',
        'module4/llm-task-decomposition',
        'module4/llm-prompt-engineering',
        'module4/multimodal-fusion',
        'module4/capstone-project',
        'module4/vla-system-verification'
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/autonomous-humanoid',
        'capstone/autonomous-humanoid-project'
      ],
    },
    {
      type: 'category',
      label: 'Hardware & Deployment',
      items: [
        'hardware-requirements',
        'jetson-deployment',
        'sensor-config',
        'robot-options'
      ],
    },
    {
      type: 'category',
      label: 'Additional Resources',
      items: [
        'sim-to-real',
        'cloud-alternatives',
        'citations-guide'
      ],
    },
    {
      type: 'category',
      label: 'Assessment & Instructor Resources',
      items: [
        'assessments/README',
        'instructor-resources/README'
      ],
    },
    {
      type: 'category',
      label: 'Templates',
      items: [
        'templates/chapter-template'
      ],
    },
  ],
};

module.exports = sidebars;