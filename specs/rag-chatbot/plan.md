# Implementation Plan: Next.js/FastAPI RAG Chatbot
**Branch**: `002-rag-chatbot-integration` | **Date**: 2025-12-13 | **Spec**: specs/rag-chatbot/spec.md
**Input**: Feature specification from spec.md

**Note**: This plan is based on the specific constraints of the project: use of `context7-Mcp`, `qdrant-mcp-server`, and OpenAI Chatbot using Nextjs
|--Phase-2 Chatbot using Next.js/
|-- ├── docs/                # Source book content (Markdown)
|-- ├── backend/            # FastAPI service (RBS.1)
|-- │   ├── main.py        # API endpoints
|-- │   └── requirements.txt
|-- ├── frontend/           # Next.js application (FES.1)
|-- │   ├── pages/
|-- │   └── components/    # ChatKit UI components (FES.3)
|-- └── ingestion/          # Python script for data processing (DIP.1)

## Summary
This plan outlines the implementation of a Retrieval-Augmented Generation (RAG) chatbot system that allows users to interact with book content through a Next.js frontend. The system uses FastAPI backend with OpenAI agents, Qdrant vector database, and ChatKit SDKs for seamless chat experience. The content is sourced from Markdown files in the `/docs` folder and processed using the `context7-Mcp` utility.

## Technical Context
* **Language/Version**: Python 3.10+, JavaScript/TypeScript (Next.js).
* **Primary Dependencies**: Next.js, React, FastAPI, OpenAI Agent SDK, ChatKit SDKs, Qdrant, `sentence-transformers`, `context7-Mcp`, `qdrant-mcp-server`.
* **Storage**: Qdrant Cloud (vector storage), Files (`/docs` Markdown).
* **Testing**: E2E testing, Contextual Query validation (RBS.4/FES.5), Ingestion verification (DIP.4), Performance testing (response time under 2 seconds).
* **Target Platform**: Next.js (Frontend), Serverless/Cloud (FastAPI Backend), Qdrant Cloud.
* **Constraints**:
  1. Must use Qdrant Cloud Free Tier (storage limitations apply)
  2. Must use `context7-Mcp` for content preprocessing
  3. Must use `qdrant-mcp-server` for indexing operations
  4. Must use OpenAI Agent SDK for RAG orchestration
  5. Must use ChatKit SDKs for chat functionality
* **Non-Functional Requirements**:
  1. Support up to 100 concurrent users with auto-scaling capability
  2. Response time under 2 seconds for typical queries
  3. Persistent Qdrant collection data between sessions
  4. Error handling with 3 retry attempts and 10s timeout for API calls
  5. Secure session management using OAuth 2.0 with JWT tokens
  6. Data retention: Indefinite storage with user consent management
  7. Fallback: Fail gracefully with user-friendly error message when Qdrant is unavailable

## Execution Plan Tables

### Phase 0: Project Setup and Environment Configuration
| Task ID | Task Description | Dependencies | Spec IDs Covered |
|---------|------------------|--------------|------------------|
| P0.T1 | Initialize project directory structure (backend/, frontend/, ingestion/) | None | FES.1, RBS.1 |
| P0.T2 | Set up development environment and install dependencies (Python, Node.js) | P0.T1 | FES.1, RBS.1 |
| P0.T3 | Configure version control with Git and initial commit | P0.T2 | - |
| P0.T4 | Set up environment variables for API keys and Qdrant credentials | P0.T2 | RBS.1, DIP.3 |

### Phase 1: Data Ingestion Pipeline (DIP)
| Task ID | Task Description | Dependencies | Spec IDs Covered |
|---------|------------------|--------------|------------------|
| P1.T1 | Implement context7-Mcp integration for document processing from /docs | P0.T2 | DIP.1 |
| P1.T2 | Set up Qdrant vector database connection via qdrant-mcp-server and create persistent `book_content` collection | P0.T4 | DIP.3, DIP.4 |
| P1.T3 | Implement embedding pipeline using all-MiniLM-L6-v2 model (384-dim) | P1.T2 | DIP.2 |
| P1.T4 | Develop document ingestion pipeline with metadata extraction including section/chapter information from context7-Mcp | P1.T1, P1.T3 | DIP.1, DIP.2, DIP.3, DIP.4 |

### Phase 2: RAG Backend Service (RBS)
| Task ID | Task Description | Dependencies | Spec IDs Covered |
|---------|------------------|--------------|------------------|
| P2.T1 | Initialize FastAPI application with OpenAI Agent SDK integration | P0.T2 | RBS.1 |
| P2.T2 | Implement ChatKit session endpoint (/api/chatkit/session) with OAuth 2.0 and JWT token support | P2.T1 | RBS.2 |
| P2.T3 | Create custom Agent Tool that interfaces with Qdrant through qdrant-mcp-server for vector database retrieval | P1.T4, P2.T1 | RBS.3 |
| P2.T4 | Implement contextual query support with selected_text parameter | P2.T3 | RBS.4 |
| P2.T5 | Add error handling with 3 retry attempts and 10s timeout for API calls | P2.T1-P2.T4 | RBS.1-RBS.4 |
| P2.T6 | Implement graceful failure mechanism when Qdrant is unavailable with user-friendly error messages | P2.T5 | Non-functional |

### Phase 3: Next.js Frontend Development (FES)
| Task ID | Task Description | Dependencies | Spec IDs Covered |
|---------|------------------|--------------|------------------|
| P3.T1 | Initialize Next.js application with ChatKit.js SDK integration | P0.T2 | FES.1, FES.3 |
| P3.T2 | Implement /AI-book endpoint to display book content | P3.T1 | FES.2 |
| P3.T3 | Integrate ChatKit React component for chat interface | P3.T1 | FES.3 |
| P3.T4 | Implement global text selection listener using JavaScript `window.getSelection().toString()` for book content sections | P3.T2 | FES.4 |
| P3.T5 | Create contextual query mechanism with payload structure: `{ message: string, selected_text: string | null }` | P3.T3, P3.T4 | FES.5 |
| P3.T6 | Configure CORS to allow communication with backend API | P2.T5, P3.T5 | FES.6 |
| P3.T7 | Optimize frontend for performance to achieve response times under 2 seconds | P3.T1-P3.T6 | Non-functional |
| P3.T8 | Implement error handling with graceful fallback when backend services are unavailable | P3.T6 | Non-functional |

### Phase 4: Integration and Finalization
| Task ID | Task Description | Dependencies | Spec IDs Covered |
|---------|------------------|--------------|------------------|
| P4.T1 | Integrate frontend with backend API endpoints | P2.T6, P3.T8 | All |
| P4.T2 | Conduct end-to-end testing of RAG functionality | P4.T1 | All |
| P4.T3 | Perform performance testing for up to 100 concurrent users with auto-scaling | P4.T2 | Non-functional |
| P4.T4 | Validate response time under 2 seconds for typical queries | P4.T3 | Non-functional |
| P4.T5 | Test error handling with 3 retry attempts and 10s timeout | P4.T2 | Non-functional |
| P4.T6 | Verify data retention with user consent management | P4.T2 | Non-functional |
| P4.T7 | Test graceful failure mechanisms when Qdrant is unavailable | P4.T2 | Non-functional |
| P4.T8 | Document setup and deployment procedures | P4.T3-P4.T7 | - |
| P4.T9 | Final validation against acceptance criteria | P4.T8 | All |