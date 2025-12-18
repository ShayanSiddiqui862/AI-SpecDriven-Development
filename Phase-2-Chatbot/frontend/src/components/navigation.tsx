'use client';
import { createAuthClient } from "better-auth/react";
import Link from 'next/link';
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

export default function Navigation() {
  const { data: session, isPending } = authClient.useSession();
  const router = useRouter();

  const handleSignOut = async () => {
    await authClient.signOut();
    localStorage.removeItem("access_token");
    window.location.reload();
    router.push('/signin');
  };

  return (
    <nav className="bg-gray-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link href="/" className="text-xl font-bold">
              AI Chatbot
            </Link>
            <div className="ml-10 flex items-baseline space-x-4">
              <Link href="/" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">
                Home
              </Link>
              <Link href="/dashboard" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">
                Dashboard
              </Link>
              <Link href="/profile" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">
                Profile
              </Link>
              <Link href="/chatkit" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">
                Chat
              </Link>
              <Link href="/AI-book" className="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">
                AI Books
              </Link>
            </div>
          </div>

          <div className="flex items-center">
            {isPending ? (
              <span className="text-sm">Loading...</span>
            ) : session ? (
              <div className="flex items-center space-x-4">
                <span className="text-sm">Welcome, {session?.user?.name || session?.user?.email}</span>
                <button
                  onClick={handleSignOut}
                  className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm"
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Link href="/signin" className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm">
                  Sign In
                </Link>
                <Link href="/signup" className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm">
                  Sign Up
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}