# RAG Chatbot Backend

This is the backend server for the RAG (Retrieval-Augmented Generation) Chatbot that integrates with Qdrant vector database.

## Features

- **RAG Query Processing**: Process user queries with context from book content
- **Qdrant Integration**: Vector storage and similarity search
- **Book Content Ingestion**: Upload PDF, TXT, and other text formats
- **Authentication**: JWT-based authentication and authorization
- **Session Management**: Chat session creation and management

## API Endpoints

### Core RAG Endpoints
- `POST /api/rag/query` - Process RAG queries with book content context
- `POST /api/content/search` - Search book content directly

### Ingestion Endpoints
- `POST /api/ingestion/upload` - Upload book content (PDF, TXT, MD)
- `POST /api/ingestion/clear-collection` - Clear all content from Qdrant
- `GET /api/ingestion/status` - Get ingestion service status

### Session Management
- `POST /api/chatkit/session` - Create chat sessions
- `GET /api/chatkit/session/{session_id}` - Get session details
- `DELETE /api/chatkit/session/{session_id}` - Delete a session

### Health Checks
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/rag/health` - RAG service health
- `GET /api/ingestion/health` - Ingestion service health

## Setup Instructions

1. **Navigate to the project directory:**
   ```bash
   cd "Phase-2 Chatbot using Nextjs"
   ```

2. **Set up the virtual environment and install dependencies:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a `.env` file in the backend directory with your configuration:**
   ```env
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_api_key
   GEMINI_API_KEY=your_gemini_api_key
   BACKEND_HOST=localhost
   BACKEND_PORT=8000
   SECRET_KEY=your_secret_key_for_jwt
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

4. **Start the Qdrant server** (if running locally):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
     -e QDRANT__SERVICE__API_KEY=your_qdrant_api_key \
     qdrant/qdrant
   ```

5. **Start the backend server:**
   ```bash
   cd "Phase-2 Chatbot using Nextjs"  # Make sure you're in this directory
   python run_backend.py
   ```

   Or alternatively:
   ```bash
   cd "Phase-2 Chatbot using Nextjs"
   python -m uvicorn backend.main:app --reload
   ```

## Using the Ingestion API

1. **Upload a book file:**
   - Open your browser to `http://localhost:8000/docs`
   - Find the `/api/ingestion/upload` endpoint
   - Use the "Try it out" feature to upload your book file

2. **Check the status:**
   - GET request to `http://localhost:8000/api/ingestion/status`

3. **Clear the collection:**
   - POST request to `http://localhost:8000/api/ingestion/clear-collection`

## Dependencies

- FastAPI
- Qdrant Client
- Sentence Transformers
- OpenAI
- LiteLLM
- PyPDF2

## Troubleshooting

- If you get import errors, make sure you're running the server from the `Phase-2 Chatbot using Nextjs` directory
- Ensure Qdrant server is running before starting the backend
- Check that all environment variables are properly set in your `.env` file