'use client';

import { useState } from 'react';
// Assuming 'better-auth/react' and its plugins handle the API calls correctly
import { createAuthClient } from "better-auth/react";
import { usernameClient } from "better-auth/client/plugins";
import { useRouter } from 'next/navigation';
import Link from 'next/link';

// Placeholder for a toast notification library (you would need to install one like 'react-hot-toast')
const showToast = (message: string, type: 'success' | 'error') => {
    // In a real app, this would trigger a visible notification
    console.log(`[${type.toUpperCase()}] ${message}`); 
};



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

  },
  plugins: [usernameClient()],
});


export default function SignInPage() {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
        const response = await authClient.signIn.username({
            username: formData.username,
            password: formData.password,
        });

        // BETTER-AUTH returns the data from your FastAPI /login endpoint here
        if (response.data && 'access_token' in response.data) {
            // MANUALLY save the token if it's not in local storage
            localStorage.setItem('access_token', String(response.data.access_token));
            
            showToast('Login successful!', 'success');
            router.push('/'); 
        } else if (response.error) {
            setError(response.error.message || 'Login failed');
        }
    } catch (err: any) {
        setError('An unexpected error occurred');
    } finally {
        setLoading(false);
    }
};

  return (
    <div className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold">
            Sign in to your account
          </h2>
        </div>

        {error && (
          <div role="alert" className="p-4 text-red-700 bg-red-100 rounded">
            <span>{String(error)}</span>
          </div>
        )}

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div>
            <label htmlFor="username" className="block text-sm font-medium">
              Username
            </label>
            <input
              id="username"
              name="username"
              type="text"
              autoComplete="username"
              required
              value={formData.username}
              onChange={handleChange}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 focus:outline-none"
              placeholder="chatbot_user"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
              value={formData.password}
              onChange={handleChange}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 focus:outline-none"
              placeholder="••••••••"
            />
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Processing...' : 'Sign in'}
            </button>
          </div>
        </form>

  

        <div className="text-center mt-4">
          <p className="text-sm text-gray-600">
            Don't have an account?{' '}
            <Link href="/signup" className="font-medium text-indigo-600 hover:text-indigo-500">
              Sign up
            </Link>
          </p>
          <Link href="/" className="text-sm mt-2 inline-block font-medium text-indigo-600 hover:text-indigo-500">
            &larr; Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}