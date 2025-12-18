'use client';

import { ReactNode } from 'react';
import { usePathname } from 'next/navigation';

interface ModuleLayoutProps {
  children: ReactNode;
}

export default function ModuleLayout({ children }: ModuleLayoutProps) {
  const pathname = usePathname();
  const moduleId = pathname.split('/')[2] || 'module1';

  const moduleTitles: Record<string, string> = {
    'module1': 'Module 1: ROS 2 Humanoid Control',
    'module2': 'Module 2: Digital Twin Environment Setup',
    'module3': 'Module 3: AI-Robot Brain Development',
    'module4': 'Module 4: Vision-Language-Action (VLA) System Integration'
  };

  const moduleTitle = moduleTitles[moduleId] || 'Module';

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold text-gray-900">{moduleTitle}</h1>
        </div>
      </header>
      <main>
        {children}
      </main>
    </div>
  );
}