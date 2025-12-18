'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { createAuthClient } from "better-auth/react";
const authClient = createAuthClient({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL,
  fetchOptions: {
    onRequest: async (context) => {
      const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
      
      if (token) {
        // Correct way to update headers to satisfy TypeScript
        context.headers = {
          ...context.headers,
          Authorization: `Bearer ${token}`,
        };
      }
      return context; // Return the full context object
    },
    onResponse: async (context) => {
      if (context.response.status === 401) {
        // Token is expired or invalid
        if (typeof window !== "undefined") {
          localStorage.removeItem("access_token");
          // Redirect to signin if we are not already there
          if (!window.location.pathname.includes('/signin')) {
             window.location.href = "/signin?error=session_expired";
          }
        }
      }
      return context;
    },
  },
});

export default function AIBOOKPage() {
  const pathname = usePathname();

  const modules = [
    {
      id: 'module1',
      title: 'Module 1: ROS 2 Humanoid Control',
      description: 'Learn to create and control simulated humanoid joints using ROS 2',
      link: '/AI-book/module1',
      icon: '🤖'
    },
    {
      id: 'module2',
      title: 'Module 2: Digital Twin Environment',
      description: 'Build high-fidelity simulated environments with Gazebo and Unity',
      link: '/AI-book/module2',
      icon: '🌍'
    },
    {
      id: 'module3',
      title: 'Module 3: AI-Robot Brain Development',
      description: 'Deploy NVIDIA Isaac Sim and implement VSLAM and Nav2 for navigation',
      link: '/AI-book/module3',
      icon: '🧠'
    },
    {
      id: 'module4',
      title: 'Module 4: Vision-Language-Action System',
      description: 'Create end-to-end systems that execute natural language commands',
      link: '/AI-book/module4',
      icon: '👁️'
    }
  ];

  return (
    <div className="py-6 sm:py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Hero Section */}
        <div className="text-center mb-10 sm:mb-16">
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4 sm:mb-6">
            Physical AI & Humanoid Robotics Textbook
          </h1>
          <p className="text-base sm:text-lg md:text-xl text-gray-600 dark:text-gray-300 max-w-2xl sm:max-w-3xl md:max-w-4xl mx-auto mb-6 sm:mb-8">
            Welcome to the comprehensive guide covering essential concepts and practical implementations
            needed to understand and develop humanoid robotics systems using modern AI techniques.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4 sm:gap-6">
            <Link
              href="/AI-book/module1"
              className="px-5 sm:px-6 py-2.5 sm:py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-lg"
            >
              Start Reading
            </Link>
            <Link
              href="/dashboard"
              className="px-5 sm:px-6 py-2.5 sm:py-3 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-colors"
            >
              Back to Dashboard
            </Link>
          </div>
        </div>

        {/* Modules Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4 sm:gap-6 mb-10 sm:mb-16">
          {modules.map((module, index) => (
            <Link
              key={module.id}
              href={module.link}
              className="block group"
            >
              <div className={`bg-white dark:bg-gray-800 rounded-lg sm:rounded-xl shadow-md sm:shadow-lg hover:shadow-lg sm:hover:shadow-xl transition-all duration-300 border border-gray-200 dark:border-gray-700 p-5 sm:p-6 md:p-8 hover:transform hover:-translate-y-1 ${pathname.includes(module.id) ? 'border-blue-500 ring-2 ring-blue-500/20' : ''}`}>
                <div className="text-3xl sm:text-4xl mb-3 sm:mb-4 group-hover:scale-105 transition-transform duration-300">
                  {module.icon}
                </div>
                <h2 className="text-xl sm:text-2xl font-bold mb-2 sm:mb-3 text-gray-800 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400">
                  {module.title}
                </h2>
                <p className="text-base sm:text-lg text-gray-600 dark:text-gray-300 mb-3 sm:mb-4">
                  {module.description}
                </p>
                <span className="inline-block px-3 sm:px-4 py-1.5 sm:py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full text-xs sm:text-sm font-medium">
                  Explore Module
                </span>
              </div>
            </Link>
          ))}
        </div>

        {/* Course Overview */}
        <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-md sm:shadow-xl p-5 sm:p-6 md:p-8 mb-8 sm:mb-12 border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-4 sm:mb-6 text-center">Course Overview</h2>
          <p className="text-base sm:text-lg text-gray-600 dark:text-gray-300 mb-4 sm:mb-6 text-center max-w-xl sm:max-w-2xl md:max-w-3xl mx-auto">
            This textbook is structured around four core modules. Each module builds upon the previous one,
            providing a comprehensive learning path from basic ROS 2 concepts to advanced VLA integration.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 sm:p-5 md:p-6 rounded-lg sm:rounded-xl border border-blue-100 dark:border-blue-800/50">
              <h3 className="text-lg sm:text-xl font-bold text-blue-800 dark:text-blue-200 mb-2 sm:mb-3">Module 1: Foundation</h3>
              <p className="text-blue-700 dark:text-blue-300 text-sm sm:text-base">Establishes foundational knowledge with ROS 2 and humanoid control</p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 sm:p-5 md:p-6 rounded-lg sm:rounded-xl border border-green-100 dark:border-green-800/50">
              <h3 className="text-lg sm:text-xl font-bold text-green-800 dark:text-green-200 mb-2 sm:mb-3">Module 2: Simulation</h3>
              <p className="text-green-700 dark:text-green-300 text-sm sm:text-base">Focuses on creating realistic simulated environments</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 sm:p-5 md:p-6 rounded-lg sm:rounded-xl border border-purple-100 dark:border-purple-800/50">
              <h3 className="text-lg sm:text-xl font-bold text-purple-800 dark:text-purple-200 mb-2 sm:mb-3">Module 3: AI Perception</h3>
              <p className="text-purple-700 dark:text-purple-300 text-sm sm:text-base">Develops AI perception and navigation capabilities</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 sm:p-5 md:p-6 rounded-lg sm:rounded-xl border border-yellow-100 dark:border-yellow-800/50">
              <h3 className="text-lg sm:text-xl font-bold text-yellow-800 dark:text-yellow-200 mb-2 sm:mb-3">Module 4: Integration</h3>
              <p className="text-yellow-700 dark:text-yellow-300 text-sm sm:text-base">Integrates all components into an end-to-end VLA system</p>
            </div>
          </div>
        </div>

        {/* Additional Resources */}
        <div className="bg-linear-to-br from-blue-600 to-purple-600 rounded-xl sm:rounded-2xl p-5 sm:p-6 md:p-8 text-white text-center">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4">Ready to Dive Deeper?</h2>
          <p className="text-blue-100 mb-4 sm:mb-6 max-w-xs sm:max-w-sm md:max-w-md mx-auto">
            Access additional resources, code examples, and interactive exercises to enhance your learning experience.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-3 sm:gap-4">
            <Link
              href="/dashboard"
              className="px-5 sm:px-6 py-2.5 sm:py-3 bg-white text-blue-600 rounded-lg font-medium hover:bg-gray-100 transition-colors"
            >
              View Dashboard
            </Link>
            <Link
              href="/chatkit"
              className="px-5 sm:px-6 py-2.5 sm:py-3 bg-transparent border-2 border-white text-white rounded-lg font-medium hover:bg-white/10 transition-colors"
            >
              AI Chat Support
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}