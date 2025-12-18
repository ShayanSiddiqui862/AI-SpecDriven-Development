// This API route proxies the registration request to the backend
import { NextRequest, NextResponse } from 'next/server';

const BACKEND_BASE_URL = "https://shayan345-backend.hf.space";

export async function POST(request: NextRequest) {
    const backendUrl = BACKEND_BASE_URL;
    const body = await request.json();

    try {
        // The backend registration endpoint
        const backendPath = '/api/auth/register'; 

        const response = await fetch(`${backendUrl}${backendPath}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        // Forward the response (including status and body) directly
        const responseData = await response.json();
        return NextResponse.json(responseData, {
            status: response.status,
            headers: response.headers,
        });

    } catch (error) {
        console.error('Registration proxy request failed:', error);
        return NextResponse.json({ detail: 'An error occurred during registration.' }, {
            status: 500,
        });
    }
}