'use client';

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
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useEffect } from 'react';

export default function DashboardPage() {
  const { data: session, isPending } = authClient.useSession();
  const router = useRouter();

  const handleSignOut = async () => {
    await authClient.signOut();
    localStorage.removeItem("access_token");
    window.location.reload();
    router.push('/auth');
  };

  if (isPending) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  if (!session) {
    return null; // Redirect happens in useEffect
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <div className="flex items-center space-x-4">
            <span className="text-gray-700">Welcome, {session?.user?.name}</span>
            <button
              onClick={handleSignOut}
              className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded"
            >
              Sign Out
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">User Information</h2>
          <div className="space-y-2">
            <p><span className="font-medium">Username:</span> {session?.user?.name}</p>
            <p><span className="font-medium">Full Name:</span> {session?.user?.name || 'Not provided'}</p>
            <p><span className="font-medium">Email:</span> {session?.user?.email}</p>
          
          </div>
        </div>

        <div className="mt-6 bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Navigation</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Link
              href="/profile"
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded text-center"
            >
              Profile
            </Link>
            <Link
              href="/chatkit"
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-3 px-4 rounded text-center"
            >
              Chat Interface
            </Link>
            <Link
              href="/AI-book"
              className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded text-center"
            >
              AI Books
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}