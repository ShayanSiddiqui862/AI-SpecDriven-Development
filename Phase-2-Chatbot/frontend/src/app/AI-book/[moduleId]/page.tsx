'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
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

export default function ModulePage() {
  const { moduleId } = useParams();
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModuleContent = async () => {
      if (moduleId && typeof moduleId === 'string') {
        try {
          const response = await fetch(`/api/module-content?id=${moduleId}`);
          if (!response.ok) {
            throw new Error(`Failed to fetch module content: ${response.statusText}`);
          }
          const data = await response.json();
          setContent(data.content);
        } catch (err) {
          console.error('Error fetching module content:', err);
          setError('Failed to load module content. Please try again later.');
          setContent(`# ${moduleId.charAt(0).toUpperCase() + moduleId.slice(1)}\n\nModule content could not be loaded. This section would contain information about ${moduleId.replace('module', 'Module ')}.`);
        } finally {
          setLoading(false);
        }
      }
    };

    fetchModuleContent();
  }, [moduleId]);

  if (loading) {
    return (
      <div className="py-6 sm:py-8">
        <div className="max-w-3xl sm:max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-lg sm:text-xl text-gray-700 dark:text-gray-300">Loading module content...</div>
        </div>
      </div>
    );
  }

  // Define modules for navigation
  const modules = [
    { id: 'module1', title: 'Module 1: ROS 2 Humanoid Control', icon: '🤖' },
    { id: 'module2', title: 'Module 2: Digital Twin Environment', icon: '🌍' },
    { id: 'module3', title: 'Module 3: AI-Robot Brain Development', icon: '🧠' },
    { id: 'module4', title: 'Module 4: Vision-Language-Action System', icon: '👁️' },
  ];

  const currentModule = modules.find(m => m.id === moduleId);
  const currentIndex = modules.findIndex(m => m.id === moduleId);
  const prevModule = currentIndex > 0 ? modules[currentIndex - 1] : null;
  const nextModule = currentIndex < modules.length - 1 ? modules[currentIndex + 1] : null;

  return (
    <div className="py-6 sm:py-8">
      <div className="max-w-3xl sm:max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Module Header */}
        <header className="mb-6 sm:mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4 mb-3 sm:mb-4">
            <span className="text-2xl sm:text-3xl">{currentModule?.icon}</span>
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">
              {currentModule?.title || 'Module'}
            </h1>
          </div>
          <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
            <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2.5 sm:px-3 py-0.5 sm:py-1 rounded-full text-xs sm:text-sm">
              {moduleId}
            </span>
          </div>
        </header>

        {/* Content Area */}
        <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-md sm:shadow-lg p-5 sm:p-6 md:p-8 mb-6 sm:mb-8 border border-gray-200 dark:border-gray-700">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              h1: ({node, ...props}) => <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-4 sm:mb-6 mt-6 sm:mt-8" {...props} />,
              h2: ({node, ...props}) => <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4 mt-5 sm:mt-6 border-b border-gray-200 dark:border-gray-700 pb-1 sm:pb-2" {...props} />,
              h3: ({node, ...props}) => <h3 className="text-lg sm:text-xl font-semibold text-gray-800 dark:text-gray-200 mb-2 sm:mb-3 mt-4" {...props} />,
              p: ({node, ...props}) => <p className="text-gray-700 dark:text-gray-300 mb-3 sm:mb-4 leading-relaxed text-sm sm:text-base" {...props} />,
              ul: ({node, ...props}) => <ul className="list-disc pl-5 sm:pl-6 mb-3 sm:mb-4 space-y-1 sm:space-y-2" {...props} />,
              ol: ({node, ...props}) => <ol className="list-decimal pl-5 sm:pl-6 mb-3 sm:mb-4 space-y-1 sm:space-y-2" {...props} />,
              li: ({node, ...props}) => <li className="text-gray-700 dark:text-gray-300 text-sm sm:text-base" {...props} />,
              a: ({node, ...props}) => <a className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline" {...props} />,
              code: ({node, ...props}) => <code className="bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md font-mono text-xs sm:text-sm" {...props} />,
              pre: ({node, ...props}) => <pre className="bg-gray-900 text-gray-100 p-3 sm:p-4 rounded-lg my-3 sm:my-4 overflow-x-auto text-xs sm:text-sm" {...props} />,
              blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-blue-500 pl-3 sm:pl-4 italic text-gray-600 dark:text-gray-400 my-3 sm:my-4 text-sm sm:text-base" {...props} />,
              img: ({node, ...props}) => <img className="rounded-lg my-3 sm:my-4 max-w-full h-auto" {...props} />,
            }}
          >
            {content}
          </ReactMarkdown>
        </div>

        {/* Navigation between modules */}
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4 sm:gap-0">
          {prevModule ? (
            <Link
              href={`/AI-book/${prevModule.id}`}
              className="flex items-center px-4 sm:px-6 py-2.5 sm:py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-900 transition-colors w-full sm:w-auto justify-center sm:justify-start"
            >
              ← <span className="ml-2">{prevModule.title}</span>
            </Link>
          ) : (
            <div className="w-full sm:w-auto"></div> // Empty div to maintain flex alignment
          )}

          {nextModule ? (
            <Link
              href={`/AI-book/${nextModule.id}`}
              className="flex items-center px-4 sm:px-6 py-2.5 sm:py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors w-full sm:w-auto justify-center sm:justify-end"
            >
              <span className="mr-2">{nextModule.title}</span> →
            </Link>
          ) : (
            <div className="w-full sm:w-auto"></div> // Empty div to maintain flex alignment
          )}
        </div>

        {error && (
          <div className="mt-6 sm:mt-8 p-3 sm:p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-700 text-red-700 dark:text-red-300 rounded-lg text-sm">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}