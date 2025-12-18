# Research: RAG Chatbot with Book Content Search

## Decision: Technology Stack
**Rationale**: Selected Next.js for frontend and FastAPI for backend based on requirements from spec.md and plan.md
- Next.js: React-based framework with excellent SEO, SSR capabilities, and strong ecosystem
- FastAPI: Python framework with automatic API documentation, type validation, and async support
- Qdrant: Vector database optimized for similarity search with good Python/JS SDKs
- OpenAI: Proven RAG capabilities and integration with vector databases
- ChatKit: Real-time chat interface SDK with React components
- context7-Mcp: Required preprocessing utility for document processing
- qdrant-mcp-server: Required interface for vector database operations

## Decision: Content Processing Pipeline
**Rationale**: Use context7-Mcp utility for preprocessing as required by spec and plan
- Preprocessing with context7-Mcp ensures standardized chunking and metadata extraction from /docs markdown files
- Sentence-transformers all-MiniLM-L6-v2 model for embedding generation (384 dimensions)
- Qdrant-mcp-server interface for vector indexing to maintain consistency with project constraints
- Implementation following DIP.1-DIP.4 requirements in the specification

## Decision: Architecture Pattern
**Rationale**: Service-oriented architecture with clear separation between data ingestion, backend, and frontend as outlined in plan.md
- Data Ingestion Pipeline (DIP): Handles document processing, embedding, and indexing
- Backend Service (RBS): Handles RAG operations, vector search, and OpenAI integration
- Frontend Service (FES): Handles UI, user interactions, and text selection
- API communication via REST/JSON with proper CORS configuration

## Implementation Research Findings

### Phase 0: Project Setup and Environment Configuration
- Initial project structure with backend/, frontend/, and ingestion/ directories
- Development environment setup with Python 3.10+ and Node.js 18+
- Environment variable configuration for API keys and Qdrant credentials
- Git version control initialization

### Phase 1: Data Ingestion Pipeline (DIP)
- context7-Mcp integration for document processing from /docs directory
- Qdrant vector database connection via qdrant-mcp-server
- Embedding pipeline using all-MiniLM-L6-v2 model (384-dim)
- Document ingestion pipeline with metadata extraction as per DIP.1-DIP.4

### Phase 2: RAG Backend Service (RBS)
- FastAPI application with OpenAI Agent SDK integration
- ChatKit session endpoint implementation (/api/chatkit/session)
- Custom Agent Tool for Qdrant vector database retrieval
- Contextual query support with selected_text parameter as per RBS.1-RBS.4

### Phase 3: Next.js Frontend Development (FES)
- Next.js application with ChatKit.js SDK integration
- /AI-book endpoint to display book content
- ChatKit React component for chat interface
- Global text selection listener for book content as per FES.1-FES.6
- Contextual query mechanism with selected_text payload

### Phase 4: Integration and Finalization
- Frontend-backend API integration
- End-to-end testing of RAG functionality
- Performance testing and optimization
- Documentation and final validation against acceptance criteria

## Additional Research Findings

### Performance Considerations:
- Qdrant Cloud Free Tier supports up to 5 collections and 1GB storage
- All-MiniLM-L6-v2 model provides good balance of speed and accuracy for embedding generation
- OpenAI function calling enables structured RAG responses with source citations
- Proper indexing strategies for efficient vector search operations

### Security Considerations:
- ChatKit session management with client_secret for secure communication
- CORS configuration to prevent unauthorized API access between frontend and backend
- Proper API key management for OpenAI and Qdrant services
- Secure handling of selected text context in API payloads

### Scalability Considerations:
- Asynchronous processing for embedding generation to handle large documents
- Caching mechanisms for frequently accessed content
- Load balancing for high-traffic scenarios
- Efficient vector database operations through qdrant-mcp-server interface