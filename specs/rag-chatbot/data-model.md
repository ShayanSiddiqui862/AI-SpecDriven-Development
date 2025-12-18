# Data Model: RAG Chatbot with Book Content Search

## Phase 1: Data Ingestion Pipeline (DIP) Models

### DocumentChunk (Processed by context7-Mcp)
- **id**: string (primary key, UUID)
- **source_file**: string (original markdown file path from /docs)
- **title**: string (title of the book/document)
- **section**: string (chapter/section name from context7-Mcp processing)
- **content**: string (chunked text content)
- **metadata**: object (additional metadata extracted by context7-Mcp)
- **embedding**: float[384] (vector embedding using all-MiniLM-L6-v2)
- **chunk_index**: integer (order of chunk in original document)
- **created_at**: datetime
- **updated_at**: datetime

## Phase 2: RAG Backend Service (RBS) Models

### ChatSession
- **id**: string (primary key, UUID)
- **client_secret**: string (ChatKit session identifier)
- **user_id**: string (optional, for tracking user sessions)
- **created_at**: datetime
- **updated_at**: datetime

### ChatMessage
- **id**: string (primary key, UUID)
- **session_id**: string (foreign key to ChatSession)
- **role**: string (user|assistant)
- **content**: string (message text)
- **selected_text**: string (optional, text selected by user in book content per FES.5/RBS.4)
- **context_chunks**: array (references to relevant DocumentChunk IDs from Qdrant)
- **query_vector**: float[384] (embedding of user query for retrieval)
- **timestamp**: datetime

## Phase 3: Next.js Frontend (FES) Models

### BookContentDisplay
- **id**: string (reference to DocumentChunk ID)
- **title**: string (book/document title)
- **section**: string (chapter/section name)
- **content**: string (formatted content for display)
- **source_file**: string (original markdown file path)
- **display_order**: integer (order for book content presentation)

## Relationships

- ChatSession (1) → (Many) ChatMessage
- DocumentChunk (Many) → (Many) ChatMessage (via context_chunks for RBS.3)
- BookContentDisplay (1) → (Many) ChatMessage (via selected_text for FES.5/RBS.4)

## Validation Rules

### DocumentChunk
- title, section, and content are required (DIP.1 requirement)
- embedding must be exactly 384-dimensional vector (DIP.2 requirement)
- source_file must be a valid path to markdown file in /docs (DIP.1 requirement)
- metadata must conform to context7-Mcp output format (DIP.1 requirement)

### ChatSession
- client_secret is required and must be unique (RBS.2 requirement)
- created_at is auto-generated

### ChatMessage
- role must be either 'user' or 'assistant'
- content is required
- session_id must reference existing ChatSession
- selected_text is optional but when present triggers contextual query logic (RBS.4/FES.5)
- context_chunks must reference valid DocumentChunk IDs

### BookContentDisplay
- title and content are required
- display_order must be a positive integer
- source_file must correspond to a valid DocumentChunk

## State Transitions

### ChatSession
- ACTIVE (default) → ARCHIVED (when session expires or is closed by user)

## Indexes

### DocumentChunk (Qdrant Collection: book_content)
- Vector index on embedding field for similarity search (DIP.4 requirement)
- Text index on section field for filtering (DIP.4 requirement)
- Text index on source_file for quick lookup (DIP.4 requirement)
- Index on chunk_index for document reconstruction

### ChatSession
- Unique index on client_secret for quick session lookup (RBS.2 requirement)

### ChatMessage
- Index on session_id for session-based queries
- Index on timestamp for chronological ordering
- Index on selected_text for contextual query analysis (FES.5/RBS.4)

### BookContentDisplay
- Index on display_order for content presentation
- Index on source_file for document grouping