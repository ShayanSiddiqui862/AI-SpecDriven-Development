"use client";
import Navigation from "@/components/navigation";
import Link from "next/link";
import { createAuthClient } from "better-auth/react";

const authClient = createAuthClient({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL || 'https://shayan345-backend.hf.space/',
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

export default function Home() {
  const { data: session, isPending } = authClient.useSession();

  // Define modules data
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
    <>
      <Navigation />
      <div className="flex min-h-screen flex-col bg-zinc-50 font-sans dark:bg-black">
        {/* Hero Banner */}
        <header className="bg-linear-to-br from-black to-gray-900 text-white py-16 px-4">
          <div className="max-w-6xl mx-auto text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-4">
              Physical AI & Humanoid Robotics
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
              A Comprehensive Guide to Teaching Physical AI & Humanoid Robotics Course
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4 mt-8">
              <Link
                href="/AI-book"
                className="px-8 py-4 bg-blue-600 text-white rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-1"
              >
                Start Reading - 5min ⏱️
              </Link>
              <Link
                href="/AI-book"
                className="px-8 py-4 bg-transparent border-2 border-white text-white rounded-lg text-lg font-semibold hover:bg-white hover:text-black transition-colors duration-300"
              >
                Explore Modules
              </Link>
            </div>
          </div>
        </header>

        <main className="grow">
          {/* Highlights Section */}
          <section className="py-16 bg-gray-50 dark:bg-gray-900">
            <div className="max-w-6xl mx-auto px-4">
              <h2 className="text-3xl md:text-4xl font-bold text-center mb-12 text-gray-800 dark:text-white">
                What You'll Learn
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg hover:shadow-xl  duration-300 border border-gray-200 dark:border-gray-700 hover:scale-105 transition-transform">
                  <h3 className="text-xl font-bold mb-4 text-blue-600 dark:text-blue-400">Hands-On Projects</h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Complete practical exercises with real code examples and implementations
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg hover:shadow-xl  duration-300 border border-gray-200 dark:border-gray-700 hover:scale-105 transition-transform">
                  <h3 className="text-xl font-bold mb-4 text-blue-600 dark:text-blue-400">Industry Standards</h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Follow best practices using ROS 2, NVIDIA Isaac Sim, and modern robotics tools
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg hover:shadow-xl  duration-300 border border-gray-200 dark:border-gray-700 hover:scale-105 transition-transform">
                  <h3 className="text-xl font-bold mb-4 text-blue-600 dark:text-blue-400">Academic Rigor</h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Over 240 references from peer-reviewed sources and conferences
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Modules Section */}
          <section className="py-16 bg-white dark:bg-black">
            <div className="max-w-6xl mx-auto px-4">
              <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 text-gray-800 dark:text-white">
                Course Modules
              </h2>
              <p className="text-center text-lg text-gray-600 dark:text-gray-400 mb-12 max-w-3xl mx-auto">
                A comprehensive learning path from basic ROS 2 concepts to advanced VLA integration
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {modules.map((module, index) => (
                  <Link
                    href={module.link}
                    key={index}
                    className="block group"
                  >
                    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200 dark:border-gray-700 p-8 hover:transform hover:-translate-y-2 hover:border-blue-500">
                      <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                        {module.icon}
                      </div>
                      <h3 className="text-xl font-bold mb-3 text-gray-800 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400">
                        {module.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        {module.description}
                      </p>
                      <span className="inline-block px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full text-sm font-medium">
                        Explore Module
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </section>

          {/* Additional content based on auth status */}
          <section className="py-16 bg-gray-50 dark:bg-gray-900">
            <div className="max-w-4xl mx-auto px-4 text-center">
              {session ? (
                <div>
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
                    Welcome back, {session?.user?.name || session?.user?.email}!
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-8">
                    Access your dashboard to continue your learning journey or explore the AI-powered chatbot.
                  </p>
                  <div className="flex flex-col sm:flex-row justify-center gap-4">
                    <Link
                      href="/dashboard"
                      className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
                    >
                      Go to Dashboard
                    </Link>
                    <Link
                      href="/AI-book"
                      className="px-6 py-3 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-colors"
                    >
                      Continue Reading
                    </Link>
                  </div>
                </div>
              ) : (
                <div>
                  <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
                    Get Started Today
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-8">
                    Sign in to access the AI-powered chatbot and explore AI books.
                  </p>
                  <div className="flex flex-col sm:flex-row justify-center gap-4">
                    <Link
                      href="/signin"
                      className="px-6 py-3 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-colors"
                    >
                      Sign In
                    </Link>
                    <Link
                      href="/signup"
                      className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
                    >
                      Sign Up
                    </Link>
                  </div>
                </div>
              )}
            </div>
          </section>
        </main>
      </div>
    </>
  );
}
