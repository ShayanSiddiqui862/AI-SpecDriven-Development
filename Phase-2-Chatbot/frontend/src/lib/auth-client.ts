// Custom authentication client that works with the existing backend
import { useState, useEffect } from 'react';

interface User {
  username: string;
  email: string;
  full_name?: string;
  disabled?: boolean;
  id?: string;
  createdAt?: string;
}

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

interface LoginCredentials {
  username: string;
  password: string;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

class AuthClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    // Check for token in localStorage on initialization
    this.token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  }

  async signIn(credentials: LoginCredentials): Promise<TokenResponse | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: credentials.username,
          password: credentials.password,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
      }

      const data: TokenResponse = await response.json();

      // Store tokens
      if (typeof window !== 'undefined') {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        this.token = data.access_token;
      }

      return data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }

  async signUp(userData: RegisterData): Promise<TokenResponse | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Registration failed');
      }

      const data: TokenResponse = await response.json();

      // Store tokens
      if (typeof window !== 'undefined') {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        this.token = data.access_token;
      }

      return data;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  }

  async signOut(): Promise<void> {
    try {
      const refreshToken = typeof window !== 'undefined' ? localStorage.getItem('refresh_token') : null;

      if (refreshToken) {
        await fetch(`${this.baseUrl}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear tokens
      if (typeof window !== 'undefined') {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        this.token = null;
      }
    }
  }

  async getCurrentUser(): Promise<User | null> {
    if (!this.token) {
      return null;
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/auth/me`, {
        headers: {
          'Authorization': `Bearer ${this.token}`,
        },
      });

      if (!response.ok) {
        // If token is invalid, clear it
        if (typeof window !== 'undefined') {
          localStorage.removeItem('access_token');
          this.token = null;
        }
        return null;
      }

      const user: User = await response.json();
      return user;
    } catch (error) {
      console.error('Get user error:', error);
      return null;
    }
  }

  // React hook for session
  useSession() {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
      const fetchUser = async () => {
        setLoading(true);
        const userData = await this.getCurrentUser();
        setUser(userData);
        setLoading(false);
      };

      fetchUser();

      // Set up token refresh interval
      const interval = setInterval(async () => {
        if (this.token) {
          const userData = await this.getCurrentUser();
          setUser(userData);
        }
      }, 5 * 60 * 1000); // Check every 5 minutes

      return () => clearInterval(interval);
    }, []);

    return { data: user, isPending: loading };
  }
}

// Create a singleton instance
const authClient = new AuthClient();

// Export the methods
export const signIn = async (type: string, credentials: any) => {
  if (type === 'credentials') {
    return await authClient.signIn({
      username: credentials.email || credentials.username,
      password: credentials.password,
    });
  }
  return null;
};

export const signUp = async (type: string, userData: any) => {
  if (type === 'email') {
    return await authClient.signUp({
      username: userData.name || userData.username,
      email: userData.email,
      password: userData.password,
      full_name: userData.name,
    });
  }
  return null;
};

export const signOut = async () => {
  await authClient.signOut();
};

export const useSession = () => {
  return authClient.useSession();
};