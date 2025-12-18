# Quickstart: RAG Chatbot with Book Content Search

## Prerequisites

- Python 3.10+ with pip
- Node.js 18+ with npm
- Qdrant Cloud account (Free Tier)
- OpenAI API key
- Git
- context7-Mcp utility installed and accessible
- qdrant-mcp-server running and accessible

## Implementation Phases

### Phase 0: Project Setup and Environment Configuration

1. Clone and initialize repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Initialize project directory structure:
```bash
mkdir -p backend frontend ingestion docs
```

3. Set up development environment:
```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn openai python-dotenv sentence-transformers

# Frontend setup
cd ../frontend
npm init -y
npm install next react react-dom @pusher/chatkit-client
```

4. Configure environment variables in both backend and frontend directories:

**Backend (.env):**
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_HOST=your_qdrant_cloud_url
CONTEXT7_MCP_PATH=path_to_context7_mcp_utility
QDRANT_MCP_SERVER_URL=url_to_qdrant_mcp_server
```

**Frontend (.env):**
```env
NEXT_PUBLIC_CHATKIT_INSTANCE_LOCATOR=your_chatkit_instance_locator
NEXT_PUBLIC_CHATKIT_KEY=your_chatkit_key
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### Phase 1: Data Ingestion Pipeline (DIP)

1. Set up ingestion dependencies:
```bash
cd ingestion
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install python-dotenv qdrant-client sentence-transformers
```

2. Process markdown files from /docs using context7-Mcp:
```bash
# Run the ingestion pipeline
python -m ingestion.main
```

This will:
- Process all markdown files in `/docs` using context7-Mcp utility
- Generate embeddings using all-MiniLM-L6-v2 model (384-dim)
- Index vectors in Qdrant via qdrant-mcp-server as per DIP.1-DIP.4

### Phase 2: RAG Backend Service (RBS)

1. Install additional backend dependencies:
```bash
cd backend
source venv/bin/activate
pip install openai-agent-sdk
```

2. Start the backend service:
```bash
uvicorn main:app --reload --port 8000
```

3. Backend provides these key endpoints:
- `POST /api/chatkit/session` - Create new chat session (RBS.2)
- `POST /api/rag/query` - RAG query with contextual support (RBS.3, RBS.4)
- `GET /api/content/search` - Search book content

### Phase 3: Next.js Frontend Development (FES)

1. Set up frontend dependencies:
```bash
cd frontend
npm install @pusher/chatkit-client next react react-dom
```

2. Create the /AI-book endpoint to display book content:
```bash
# Create pages/AI-book.js
mkdir -p pages
```

3. Start the frontend:
```bash
npm run dev
```

### Phase 4: Integration and Finalization

1. Run both services:
```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

2. Verify CORS configuration between frontend and backend

3. Test the complete RAG functionality with contextual queries

## API Endpoints

### Backend (http://localhost:8000)

- `POST /api/chatkit/session` - Create new chat session (RBS.2)
- `POST /api/rag/query` - RAG query with selected_text parameter support (RBS.3, RBS.4)
- `GET /api/content/search` - Search book content

### Frontend (http://localhost:3000)

- `/` - Main chat interface with book content
- `/AI-book` - Book content display (FES.2)
- `/chat` - Standalone chat interface

## Usage

1. Navigate to `http://localhost:3000`
2. Book content will be displayed on the `/AI-book` page (FES.2)
3. Select text in the book content to provide context (FES.4)
4. Ask questions in the chat interface
5. The RAG system will use both your query and selected text as context (FES.5, RBS.4)

## Development Commands

### Backend
```bash
# Run tests
cd backend
pytest tests/

# Format code
black src/
```

### Frontend
```bash
# Run tests
cd frontend
npm test

# Build for production
npm run build
```

### Ingestion Pipeline
```bash
# Run ingestion tests
cd ingestion
python -m pytest tests/

# Re-process all documents
python -m ingestion.main --force-reprocess
```