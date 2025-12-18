// This API route proxies a GET request to the backend (e.g., for user session data)
import { NextRequest, NextResponse } from 'next/server';

// Assuming this handles the backend URL (e.g., http://localhost:8000)
const BACKEND_BASE_URL = "https://shayan345-backend.hf.space";

// Helper function to extract auth headers (cookies/Authorization) from the incoming request
function getAuthHeaders(request: NextRequest): Record<string, string> {
    const authorization = request.headers.get('authorization');
    const cookie = request.headers.get('cookie');

    const headers: Record<string, string> = {};
    if (authorization) {
        headers['Authorization'] = authorization;
    }
    // CRITICAL: Forward cookies so the backend can identify the session
    if (cookie) {
        headers['Cookie'] = cookie;
    }
    return headers;
}


// --- GET HANDLER ---
export async function GET(request: NextRequest) {
    const backendUrl = BACKEND_BASE_URL;
    
    // Define the specific backend endpoint you want to proxy for this GET request
    // Example: Fetching the current authenticated user's details
    const backendPath = '/api/auth/login'; 

    try {
        const response = await fetch(`${backendUrl}${backendPath}`, {
            method: 'GET',
            // CRITICAL: Forward cookies/auth headers from the client to the backend
            headers: getAuthHeaders(request), 
            // Query parameters (search params) are automatically included if you use request.url
        });

        // 1. Get response data
        const responseData = await response.json();
        
        // 2. Forward all headers from the backend response (CRITICAL for setting new cookies)
        const responseHeaders = new Headers(response.headers);
        responseHeaders.set('Content-Type', 'application/json');

        return NextResponse.json(responseData, {
            status: response.status,
            headers: responseHeaders,
        });

    } catch (error) {
        console.error('GET proxy request failed:', error);
        return NextResponse.json({ detail: 'An internal error occurred while fetching data.' }, {
            status: 500,
        });
    }
}