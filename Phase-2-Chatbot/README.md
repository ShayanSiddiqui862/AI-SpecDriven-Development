# RAG Chatbot Implementation

This project implements a Retrieval-Augmented Generation (RAG) chatbot system that allows users to interact with book content through a Next.js frontend. The system uses FastAPI backend with OpenAI agents, Qdrant vector database, and ChatKit SDKs for seamless chat experience.

## Features Implemented

### Phase 1: Setup
- ✅ Created project structure (backend/, frontend/, ingestion/)
- ✅ Initialized Python project with required dependencies
- ✅ Initialized Next.js project with ChatKit.js SDK dependencies
- ✅ Configured environment variables for API keys and Qdrant credentials

### Phase 2: Foundational
- ✅ Set up Qdrant vector database connection via qdrant-mcp-server
- ✅ Configured CORS middleware for Next.js frontend communication
- ✅ Implemented error handling with 3 retry attempts and 10s timeout
- ✅ Implemented OAuth 2.0 and JWT token framework for secure session management
- ✅ Integrated Neon database for user credentials and chat session storage

### Phase 3: Data Ingestion Pipeline
- ✅ Implemented context7-Mcp integration for document processing
- ✅ Created embedding pipeline using sentence-transformer/all-MiniLM-L6-V2 model
- ✅ Implemented Hugging Face model download function with caching
- ✅ Created model caching mechanism for faster subsequent loads
- ✅ Developed document ingestion pipeline with metadata extraction
- ✅ Implemented Qdrant indexing via qdrant-mcp-server with proper metadata
- ✅ Added validation for 384-dimensional embeddings
- ✅ Created command-line interface for ingestion pipeline

### Phase 4: RAG Backend Service
- ✅ Initialized FastAPI application with OpenAI Agent SDK integration
- ✅ Implemented ChatKit session endpoint with OAuth 2.0 and JWT support
- ✅ Created custom Agent Tool that interfaces with Qdrant through qdrant-mcp-server
- ✅ Implemented contextual query endpoint with selected_text parameter
- ✅ Added error handling with 3 retry attempts and 10s timeout
- ✅ Implemented graceful failure mechanism when Qdrant is unavailable

### Phase 5: Next.js Frontend
- ✅ Initialized Next.js application with ChatKit.js SDK integration
- ✅ Implemented /AI-book endpoint to display book content
- ✅ Integrated ChatKit React component for chat interface
- ✅ Implemented global text selection listener using JavaScript
- ✅ Created contextual query mechanism with proper payload structure
- ✅ Configured CORS for communication with backend API
- ✅ Implemented error handling with graceful fallback

### Phase 6: API Contract Implementation
- ✅ Implemented /api/chatkit/session POST endpoint
- ✅ Implemented /api/rag/query POST endpoint with contextual support
- ✅ Implemented /api/content/search GET endpoint
- ✅ Added request/response validation using Pydantic models
- ✅ Implemented proper error responses per contract specification

## Architecture

### Backend Components
- **FastAPI Application**: Main backend service with OpenAI Agent SDK integration
- **Qdrant Service**: Vector database connection and management
- **RAG Tools**: Custom tools for vector search and content retrieval
- **Authentication**: OAuth 2.0 and JWT token management
- **Neon DB Service**: PostgreSQL database for user credentials and chat session storage
- **Error Handling**: Comprehensive error handling with retry logic

### Frontend Components
- **Next.js Application**: Main user interface with book content display
- **Chat Interface**: Integrated ChatKit component for conversation
- **Text Selection**: Global listener for capturing selected text
- **Better Auth**: Authentication components with Neon DB integration
- **API Service**: Service layer for backend communication

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Qdrant Cloud account (Free Tier)

### Backend Setup
1. Navigate to the backend directory: `cd Phase-2 Chatbot using Nextjs/backend`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Set environment variables in `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_URL=your_qdrant_url
   ```
4. Run the backend: `uvicorn main:app --reload`

### Frontend Setup
1. Navigate to the frontend directory: `cd Phase-2 Chatbot using Nextjs/frontend`
2. Install dependencies: `npm install`
3. Run the development server: `npm run dev`

### Ingestion Pipeline
1. Run the ingestion pipeline to process documents: `python -m ingestion.cli run`
2. This will process documents from the `docs/` directory and index them in Qdrant

## Environment Variables

### Backend (.env)
- `OPENAI_API_KEY`: OpenAI API key
- `QDRANT_API_KEY`: Qdrant API key
- `QDRANT_URL`: Qdrant server URL
- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `NEON_DB_URL`: Neon database connection string
- `BACKEND_HOST`: Backend host (default: localhost)
- `BACKEND_PORT`: Backend port (default: 8000)

### Frontend (.env.local)
- `NEXT_PUBLIC_API_URL`: Backend API URL (e.g., http://localhost:8000/api)
- `NEXT_PUBLIC_BACKEND_URL`: Backend URL (e.g., http://localhost:8000)

## API Endpoints

### Authentication
- `POST /api/chatkit/session` - Create chat session

### RAG Services
- `POST /api/rag/query` - Query the RAG system with context
- `POST /api/content/search` - Search content in the knowledge base

### Health Check
- `GET /health` - Health check for the backend service

## Usage

1. Start both backend and frontend services
2. Access the application at `http://localhost:3000`
3. Navigate to the AI Book section to view book content
4. Select text in the book content and ask questions in the chat interface
5. The AI will use the selected text as context for its responses

## Key Implementation Details

- **Embedding Model**: Uses `all-MiniLM-L6-v2` model producing 384-dimensional vectors
- **Vector Database**: Qdrant Cloud with persistent collection named `book_content`
- **Text Processing**: Uses context7-Mcp utility for document processing and chunking
- **Security**: OAuth 2.0 with JWT tokens for session management
- **Error Handling**: 3 retry attempts with 10-second timeout and exponential backoff
- **CORS**: Configured to allow communication between frontend and backend
- **Fallback**: Graceful degradation when Qdrant is unavailable

## Testing

Run the basic tests with: `python backend/test_rag.py`

## Deployment

### Backend
The backend can be deployed to platforms like Render using the provided configuration.

### Frontend
The Next.js frontend can be deployed to Vercel or other platforms supporting Next.js applications.

## Project Structure

```
Phase-2 Chatbot using Nextjs/
├── backend/
│   ├── src/
│   │   ├── auth.py          # Authentication implementation
│   │   ├── cors_config.py   # CORS configuration
│   │   ├── error_handler.py # Error handling utilities
│   │   ├── qdrant_service.py # Qdrant service
│   │   └── authentication.py # JWT and user authentication
│   ├── src/db/
│   │   └── neon_service.py   # Neon database service
│   ├── src/utils/
│   │   └── database.py       # Database utilities
│   ├── api/
│   │   ├── rag.py          # RAG endpoints
│   │   ├── sessions.py     # Session endpoints
│   │   ├── search.py       # Search endpoints
│   │   └── auth.py         # Authentication API endpoints
│   ├── tools/
│   │   └── rag_tool.py     # RAG tool implementation
│   ├── schemas/
│   │   └── rag.py          # Pydantic schemas
│   ├── exceptions/
│   │   └── __init__.py     # API exception definitions
│   ├── main.py             # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   ├── DB_SETUP.md         # Database setup documentation
│   └── test_rag.py         # Basic tests
├── frontend/
│   ├── pages/
│   │   ├── index.js        # Home page
│   │   └── AI-book.js      # Book content page
│   ├── components/
│   │   ├── ChatInterface.js # Chat interface component
│   │   ├── Layout.js       # Layout component
│   │   └── ErrorBoundary.js # Error boundary component
│   ├── hooks/
│   │   └── useTextSelection.js # Text selection hook
│   ├── services/
│   │   └── chatService.js  # Chat service API
│   ├── config/
│   │   └── api.js          # API configuration
│   ├── styles/
│   └── package.json        # Node.js dependencies
├── ingestion/
│   ├── main.py             # Main ingestion pipeline
│   ├── context7_processor.py # Context7 processor
│   ├── embedding.py        # Embedding service
│   ├── qdrant_uploader.py  # Qdrant uploader
│   ├── model_downloader.py # Model downloader
│   ├── model_cache.py      # Model caching
│   └── cli.py              # Command-line interface
├── docs/                   # Book content in Markdown format
└── .env                    # Environment variables
```