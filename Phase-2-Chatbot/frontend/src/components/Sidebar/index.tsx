'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

// Define the sidebar item types
type SidebarItem = {
  type?: 'doc' | 'category';
  label?: string;
  items?: SidebarItem[];
  href?: string;
  path?: string;
};

// Define the sidebar structure based on the sample
const sidebarStructure: SidebarItem[] = [
  {
    type: 'doc',
    label: 'Introduction',
    href: '/AI-book',
  },
  {
    type: 'category',
    label: 'Module 1 - ROS 2 Humanoid Control',
    items: [
      { type: 'doc', label: 'Overview', href: '/AI-book/module1' },
      { type: 'doc', label: 'ROS 2 Installation', href: '/AI-book/module1/ros2-installation' },
      { type: 'doc', label: 'RCLPY Agents', href: '/AI-book/module1/rclpy-agents' },
      { type: 'doc', label: 'URDF Humanoids', href: '/AI-book/module1/urdf-humanoids' },
      { type: 'doc', label: 'Bipedal Control', href: '/AI-book/module1/bipedal-control' },
      { type: 'doc', label: 'Verification', href: '/AI-book/module1/verification' },
    ],
  },
  {
    type: 'category',
    label: 'Module 2 - Digital Twin Environment Setup',
    items: [
      { type: 'doc', label: 'Overview', href: '/AI-book/module2' },
      { type: 'doc', label: 'Gazebo Worlds', href: '/AI-book/module2/gazebo-worlds' },
      { type: 'doc', label: 'URDF to SDF', href: '/AI-book/module2/urdf-to-sdf' },
      { type: 'doc', label: 'Sensor Simulation', href: '/AI-book/module2/sensor-simulation' },
      { type: 'doc', label: 'Unity HRI', href: '/AI-book/module2/unity-hri' },
      { type: 'doc', label: 'Verification', href: '/AI-book/module2/verification' },
    ],
  },
  {
    type: 'category',
    label: 'Module 3 - AI-Robot Brain Development',
    items: [
      { type: 'doc', label: 'Overview', href: '/AI-book/module3' },
      { type: 'doc', label: 'Isaac Sim Deployment', href: '/AI-book/module3/isaac-sim-deployment' },
      { type: 'doc', label: 'Synthetic Data', href: '/AI-book/module3/synthetic-data' },
      { type: 'doc', label: 'Synthetic Data Generation', href: '/AI-book/module3/synthetic-data-generation' },
      { type: 'doc', label: 'VSLAM Realsense', href: '/AI-book/module3/vslam-realsense' },
      { type: 'doc', label: 'VSLAM Implementation', href: '/AI-book/module3/vslam-implementation' },
      { type: 'doc', label: 'Nav2 Bipedal', href: '/AI-book/module3/nav2-bipedal' },
      { type: 'doc', label: 'Nav2 Bipedal Config', href: '/AI-book/module3/nav2-bipedal-config' },
      { type: 'doc', label: 'Perception Navigation Validation', href: '/AI-book/module3/perception-navigation-validation' },
    ],
  },
  {
    type: 'category',
    label: 'Module 4 - Vision-Language-Action System',
    items: [
      { type: 'doc', label: 'Overview', href: '/AI-book/module4' },
      { type: 'doc', label: 'Whisper ROS2', href: '/AI-book/module4/whisper-ros2' },
      { type: 'doc', label: 'LLM Task Decomposition', href: '/AI-book/module4/llm-task-decomposition' },
      { type: 'doc', label: 'LLM Prompt Engineering', href: '/AI-book/module4/llm-prompt-engineering' },
      { type: 'doc', label: 'Multimodal Fusion', href: '/AI-book/module4/multimodal-fusion' },
      { type: 'doc', label: 'Capstone Project', href: '/AI-book/module4/capstone-project' },
      { type: 'doc', label: 'VLA System Verification', href: '/AI-book/module4/vla-system-verification' },
    ],
  },
  {
    type: 'category',
    label: 'Capstone Project',
    items: [
      { type: 'doc', label: 'Autonomous Humanoid', href: '/AI-book/capstone/autonomous-humanoid' },
      { type: 'doc', label: 'Autonomous Humanoid Project', href: '/AI-book/capstone/autonomous-humanoid-project' },
    ],
  },
  {
    type: 'category',
    label: 'Hardware & Deployment',
    items: [
      { type: 'doc', label: 'Hardware Requirements', href: '/AI-book/hardware-requirements' },
      { type: 'doc', label: 'Jetson Deployment', href: '/AI-book/jetson-deployment' },
      { type: 'doc', label: 'Sensor Config', href: '/AI-book/sensor-config' },
      { type: 'doc', label: 'Robot Options', href: '/AI-book/robot-options' },
    ],
  },
  {
    type: 'category',
    label: 'Additional Resources',
    items: [
      { type: 'doc', label: 'Sim to Real', href: '/AI-book/sim-to-real' },
      { type: 'doc', label: 'Cloud Alternatives', href: '/AI-book/cloud-alternatives' },
      { type: 'doc', label: 'Citations Guide', href: '/AI-book/citations-guide' },
    ],
  },
];

interface SidebarItemProps {
  item: SidebarItem;
  currentPath: string;
  level?: number;
}

const SidebarItemComponent = ({ item, currentPath, level = 0 }: SidebarItemProps) => {
  const isCurrent = item.href === currentPath;
  const isParentActive = item.items?.some(subItem => subItem.href === currentPath);

  const [isExpanded, setIsExpanded] = useState(isCurrent || isParentActive);

  if (item.type === 'category') {
    return (
      <div className={`ml-${level > 0 ? 4 : 0}`}>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className={`w-full text-left py-2 px-3 rounded-md transition-colors flex justify-between items-center ${
            isCurrent || isParentActive
              ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
              : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
          }`}
        >
          <span className="font-medium">{item.label}</span>
          <span className={`transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </button>
        {isExpanded && item.items && (
          <div className="mt-1">
            {item.items.map((subItem, index) => (
              <div key={index} className="ml-4">
                <SidebarItemComponent
                  item={subItem}
                  currentPath={currentPath}
                  level={level + 1}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <Link
      href={item.href || '#'}
      className={`block py-2 px-3 rounded-md transition-colors ${
        level > 0 ? 'ml-6' : ''
      } ${
        isCurrent
          ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 font-semibold'
          : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
      }`}
    >
      {item.label}
    </Link>
  );
};

interface SidebarProps {
  onClose?: () => void;
}

const Sidebar = ({ onClose }: SidebarProps) => {
  const pathname = usePathname();

  return (
    <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:shadow-none lg:w-64">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-bold text-gray-800 dark:text-white">Course Navigation</h2>
        </div>

        {/* Navigation Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <nav>
            <ul>
              {sidebarStructure.map((item, index) => (
                <li key={index} className="mb-2">
                  <SidebarItemComponent item={item} currentPath={pathname} />
                </li>
              ))}
            </ul>
          </nav>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 text-sm text-gray-500 dark:text-gray-400">
          <p>Physical AI & Humanoid Robotics</p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;