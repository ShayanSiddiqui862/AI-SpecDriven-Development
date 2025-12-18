'use client';

import { ReactNode } from 'react';
import { createAuthClient } from 'better-auth/client';

interface AuthProviderProps {
  children: ReactNode;
}

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

export function AuthProvider({ children }: AuthProviderProps) {
  return <>{children}</>;
}

export { authClient };