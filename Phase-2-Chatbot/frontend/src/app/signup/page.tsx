'use client';

import { useState } from 'react';
// Note: Assuming you have fixed the type issues in better-auth/react imports
import { createAuthClient } from "better-auth/react"; 
import { usernameClient } from "better-auth/client/plugins";
import { useRouter } from 'next/navigation';
import Link from 'next/link';

const authClient = createAuthClient({
  baseURL:  'https://shayan345-backend.hf.space/api/register',
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


export default function SignUpPage() {
 const [formData, setFormData] = useState({
 email: '', // Used for username
 password: '',
 name: '', // Used for full_name
  username: '',
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
 // 💡 FIX: Register user by calling the local Next.js proxy route /api/register
 const response = await fetch(`/api/register`, {
 method: 'POST',
 headers: { 'Content-Type': 'application/json',
},
body: JSON.stringify({

username: formData.username,
email: formData.email, // 💡 FIX: Send the required 'email' field
password: formData.password,
 full_name: formData.name,
 }),
});

if (response.ok) {

 const signInResponse = await authClient.signIn.username({
  username: formData.username,
 password: formData.password,
 });

if (signInResponse.error) {
 router.push('/signin');
 } else {
 
setError('Login after registration failed');
 }
 } else {

 const errorData = await response.json();
 

 let errorMessage = 'Registration failed.';
 if (errorData.detail) {
 if (Array.isArray(errorData.detail)) {

 errorMessage = errorData.detail.map((d: any) => 
 `${d.loc.slice(-1)}: ${d.msg}`
 ).join('; ');
 } else {

 errorMessage = errorData.detail;
 }
 } else if (typeof errorData === 'object') {
             // Fallback for non-standard error structures
             errorMessage = JSON.stringify(errorData);
        }
 setError(errorMessage);
 }
 } catch (err: any) {

 let errorMessage = 'An unexpected network error occurred.';
      if (err.message) {
          errorMessage = err.message;
      } else if (typeof err === 'object') {
          errorMessage = JSON.stringify(err);
      }
setError(errorMessage);
 console.error(err);
 } finally {
 setLoading(false);
 }
 };

return (
 <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
 <div className="max-w-md w-full space-y-8">
 <div>
<h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
 Create a new account
 </h2>
 </div>

 {error && (
<div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
 
 <span className="block sm:inline">{String(error)}</span>
 </div>
 )}

 <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
 <div>
<label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
Full Name
 </label>
 <input
 id="name"
 name="name"
 type="text"
 required
 value={formData.name}
 onChange={handleChange}
 className="mt-1 block w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white text-gray-900 placeholder-gray-500"
 placeholder="John Doe"
 />
 </div>

 <div>
 <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-1">
 Username
 </label>
 <input
 id="username"
 name="username"
 type="text"
 required
 value={formData.username}
onChange={handleChange}
 className="mt-1 block w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white text-gray-900 placeholder-gray-500"
 placeholder="username"
 />
 </div>
  <div>
 <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
 Email Address
 </label>
 <input
 id="email_address"
 name="email"
 type="text"
 required
 value={formData.email}
onChange={handleChange}
 className="mt-1 block w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white text-gray-900 placeholder-gray-500"
 placeholder="example@email.com"
 />
 </div>

<div>
 <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
Password
 </label>
 <input
 id="password"
 name="password"
type="password"
 required
 value={formData.password}
 onChange={handleChange}
 className="mt-1 block w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white text-gray-900 placeholder-gray-500"
 placeholder="••••••••"
 />
 </div>

 <div>
 <button
 type="submit"
 disabled={loading}
 className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
 >
 {loading ? 'Processing...' : 'Sign up'}
 </button>
 </div>
 </form>

 <div className="text-center mt-4">
 <p className="text-sm text-gray-600">
 Already have an account?{' '}
 <Link href="/signin" className="font-medium text-indigo-600 hover:text-indigo-500">
 Sign in
 </Link>
</p>
 <Link href="/" className="text-indigo-600 hover:text-indigo-500 text-sm mt-2 inline-block">
 &larr; Back to home
</Link>
 </div>
 </div>
 </div> );
}