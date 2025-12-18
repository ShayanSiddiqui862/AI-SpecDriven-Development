'use client';

import { useState, useRef, useEffect } from 'react';
import MarkdownRenderer from '@/components/MarkdownRenderer';


import { useTextSelection } from '@/hooks/useTextSelection';
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

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI assistant for the Physical AI & Humanoid Robotics course. How can I help you today?',
      role: 'assistant',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([
    'Explain ROS 2 concepts',
    'How to set up Gazebo simulation?',
    'What is VSLAM?',
    'Tell me about module 1'
  ]);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');
  const selectedText = useTextSelection();
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const { data: session } = authClient.useSession();

  // Check backend connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('https://shayan345-backend.hf.space/health');
        if (response.ok) {
          setConnectionStatus('connected');
        } else {
          setConnectionStatus('disconnected');
        }
      } catch (error) {
        setConnectionStatus('disconnected');
      }
    };

    checkConnection();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send message to backend API which connects to Qdrant and OpenAI
      const response = await fetch(`https://shayan345-backend.hf.space/api/rag/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          message: inputValue,
          selected_text: selectedText || null
        }),
      });

      if (!response.ok) {
        // Handle different error status codes
        if (response.status === 503) {
          // Backend service unavailable - Qdrant might be down
          const serviceDownMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: 'The backend service is temporarily unavailable. The Qdrant vector database might be down. Please try again later.',
            role: 'assistant',
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, serviceDownMessage]);
        } else if (response.status === 429) {
          // Rate limited
          const rateLimitMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: 'You\'ve reached the rate limit. Please wait a moment before sending another message.',
            role: 'assistant',
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, rateLimitMessage]);
        } else {
          // Generic error
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: `Sorry, I encountered an error processing your message (Status: ${response.status}). Please try again.`,
            role: 'assistant',
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, errorMessage]);
        }
        return;
      }

      const data = await response.json();

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response || 'No response received from the AI assistant.',
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botResponse]);
    } catch (error: any) {
      console.error('Error sending message:', error);

      // Network error or other client-side error
      let errorMessageContent = 'Sorry, I encountered an error processing your message. Please check your connection and try again.';

      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        // Network error
        errorMessageContent = 'Unable to connect to the backend service. Please check your network connection.';
        setConnectionStatus('disconnected');
      } else if (error.message) {
        errorMessageContent = `Error: ${error.message}`;
      }

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: errorMessageContent,
        role: 'assistant',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
  };

  if (!session) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Please sign in to access the chat</h2>
          <a href="/auth" className="text-indigo-600 hover:text-indigo-800 font-medium">
            Go to Sign In
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-xl font-bold text-gray-900">AI Chat Assistant</h1>
          <div className="flex items-center space-x-4">
            <span className="text-gray-700">Welcome, {session?.user?.name}</span>
            <div className={`h-3 w-3 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500' :
              connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
            }`}></div>
            <span className="text-sm">
              {connectionStatus === 'connected' ? 'Connected' :
               connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Chat Container */}
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full py-6 px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-lg flex-1 flex flex-col h-[calc(100vh-200px)]">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {connectionStatus === 'disconnected' && (
              <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
                <strong className="font-bold">Warning: </strong>
                <span className="block sm:inline">Disconnected from backend. Some features may not work properly.</span>
              </div>
            )}
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    message.role === 'user'
                      ? 'bg-indigo-500 text-white'
                      : 'bg-gray-200 text-gray-800'
                  }`}
                >
                  {message.role === 'assistant' ? (
                    <MarkdownRenderer content={message.content} />
                  ) : (
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  )}
                  <div className={`text-xs mt-1 ${message.role === 'user' ? 'text-indigo-200' : 'text-gray-500'}`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-200 text-gray-800 rounded-lg px-4 py-2 max-w-[80%]">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Suggestions */}
          {messages.length === 1 && (
            <div className="px-4 py-2 border-b">
              <p className="text-sm text-gray-600 mb-2">Try asking about:</p>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="bg-blue-100 hover:bg-blue-200 text-blue-800 text-xs font-medium px-3 py-1.5 rounded-full transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <form onSubmit={handleSendMessage} className="border-t p-4">
            <div className="flex space-x-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about ROS, Gazebo, Isaac Sim, VSLAM, Nav2, VLA systems..."
                className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                disabled={isLoading || connectionStatus === 'disconnected'}
              />
              <button
                type="submit"
                disabled={isLoading || !inputValue.trim() || connectionStatus === 'disconnected'}
                className="bg-indigo-500 text-white rounded-lg px-4 py-2 hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
            </div>
          </form>
        </div>

        {/* Info Box */}
        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-medium text-blue-800">About this Chat</h3>
          <p className="text-blue-700 text-sm mt-1">
            This AI assistant can help you with questions about the Physical AI & Humanoid Robotics course.
            You can ask about specific modules, concepts, or implementation details.
          </p>
          {selectedText && (
            <div className="mt-2 p-2 bg-yellow-100 border border-yellow-300 rounded text-sm text-yellow-800">
              <strong>Context:</strong> Selected text: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"
            </div>
          )}
        </div>
      </div>
    </div>
  );
}