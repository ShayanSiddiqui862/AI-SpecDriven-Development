---
description: "Task list for Next.js/FastAPI RAG Chatbot implementation"
---

# Tasks: Next.js/FastAPI RAG Chatbot

**Input**: Design documents from `/specs/rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `Phase-2 Chatbot using Nextjs/backend/src/`, `Phase-2 Chatbot using Nextjs/frontend/src/`
- **Data Ingestion**: `Phase-2 Chatbot using Nextjs/ingestion/`
- **Book Content**: `Phase-2 Chatbot using Nextjs/docs/`
- Paths shown below based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure (Phase-2 Chatbot using Nextjs/backend/, Phase-2 Chatbot using Nextjs/frontend/, Phase-2 Chatbot using Nextjs/ingestion/) per implementation plan
- [ ] T002 [P] Initialize Python project with FastAPI, OpenAI Agent SDK, sentence-transformers, transformers, torch dependencies in Phase-2 Chatbot using Nextjs/backend/requirements.txt
- [ ] T003 [P] Initialize Next.js project with ChatKit.js SDK dependencies in Phase-2 Chatbot using Nextjs/frontend/package.json
- [ ] T004 [P] Configure environment variables for API keys and Qdrant credentials in Phase-2 Chatbot using Nextjs/.env files

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Set up Qdrant vector database connection via qdrant-mcp-server and create persistent `book_content` collection
- [ ] T006 Configure CORS middleware in FastAPI to allow Next.js frontend communication
- [ ] T007 [P] Set up error handling infrastructure with 3 retry attempts and 10s timeout configuration
- [ ] T008 [P] Implement OAuth 2.0 and JWT token framework for secure session management

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Data Ingestion Pipeline (Priority: P1) 🎯 MVP

**Goal**: Process Markdown files from `Phase-2 Chatbot using Nextjs/docs/` using context7-Mcp utility, generate embeddings with sentence-transformer/all-MiniLM-L6-V2 model downloaded from Hugging Face, and index in Qdrant via qdrant-mcp-server

**Independent Test**: Verify Markdown files are processed, embedded, and stored in Qdrant collection

### Implementation for User Story 1

- [ ] T009 [P] [US1] Implement context7-Mcp integration for document processing from Phase-2 Chatbot using Nextjs/docs/ in Phase-2 Chatbot using Nextjs/ingestion/context7_processor.py
- [ ] T010 [P] [US1] Create embedding pipeline using sentence-transformer/all-MiniLM-L6-V2 model (384-dim) with download from Hugging Face in Phase-2 Chatbot using Nextjs/ingestion/embedding.py
- [ ] T010a [P] [US1] Implement Hugging Face model download function to fetch sentence-transformer/all-MiniLM-L6-V2 model with caching in Phase-2 Chatbot using Nextjs/ingestion/model_downloader.py
- [ ] T010b [P] [US1] Implement model caching mechanism to store sentence-transformer/all-MiniLM-L6-V2 locally for faster subsequent loads in Phase-2 Chatbot using Nextjs/ingestion/model_cache.py
- [ ] T011 [US1] Develop document ingestion pipeline with metadata extraction including section/chapter information from context7-Mcp in Phase-2 Chatbot using Nextjs/ingestion/main.py
- [ ] T012 [US1] Implement Qdrant indexing via qdrant-mcp-server with proper metadata in Phase-2 Chatbot using Nextjs/ingestion/qdrant_uploader.py
- [ ] T013 [US1] Add validation to ensure embedding is exactly 384-dimensional vector
- [ ] T014 [US1] Create command-line interface for ingestion pipeline in Phase-2 Chatbot using Nextjs/ingestion/cli.py

**Checkpoint**: At this point, Data Ingestion Pipeline should be fully functional and testable independently

---

## Phase 4: User Story 2 - RAG Backend Service (Priority: P2)

**Goal**: Implement FastAPI backend with OpenAI Agent SDK, ChatKit session endpoint, custom RAG tool, and contextual query support

**Independent Test**: Verify API endpoints work and can perform RAG queries with contextual support

### Implementation for User Story 2

- [ ] T015 [US2] Initialize FastAPI application with OpenAI Agent SDK integration in Phase-2 Chatbot using Nextjs/backend/main.py
- [ ] T016 [US2] Implement ChatKit session endpoint (/api/chatkit/session) with OAuth 2.0 and JWT token support in Phase-2 Chatbot using Nextjs/backend/api/sessions.py
- [ ] T017 [US2] Create custom Agent Tool that interfaces with Qdrant through qdrant-mcp-server for vector database retrieval in Phase-2 Chatbot using Nextjs/backend/tools/rag_tool.py
- [ ] T018 [US2] Implement contextual query endpoint with selected_text parameter in Phase-2 Chatbot using Nextjs/backend/api/rag.py
- [ ] T019 [US2] Add error handling with 3 retry attempts and 10s timeout for API calls in Phase-2 Chatbot using Nextjs/backend/middleware/error_handler.py
- [ ] T020 [US2] Implement graceful failure mechanism when Qdrant is unavailable with user-friendly error messages in Phase-2 Chatbot using Nextjs/backend/services/fallback_handler.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Next.js Frontend (Priority: P3)

**Goal**: Create Next.js frontend with book content display, ChatKit integration, text selection, and contextual query mechanism

**Independent Test**: Verify frontend displays book content, captures text selection, and sends contextual queries

### Implementation for User Story 3

- [ ] T021 [P] [US3] Initialize Next.js application with ChatKit.js SDK integration in Phase-2 Chatbot using Nextjs/frontend/pages/_app.js
- [ ] T022 [US3] Implement /AI-book endpoint to display book content from API in Phase-2 Chatbot using Nextjs/frontend/pages/AI-book.js
- [ ] T023 [US3] Integrate ChatKit React component for chat interface in Phase-2 Chatbot using Nextjs/frontend/components/ChatInterface.js
- [ ] T024 [US3] Implement global text selection listener using JavaScript `window.getSelection().toString()` for book content sections in Phase-2 Chatbot using Nextjs/frontend/hooks/useTextSelection.js
- [ ] T025 [US3] Create contextual query mechanism with payload structure: `{ message: string, selected_text: string | null }` in Phase-2 Chatbot using Nextjs/frontend/services/chatService.js
- [ ] T026 [US3] Configure CORS to allow communication with backend API and optimize for performance in Phase-2 Chatbot using Nextjs/frontend/config/api.js
- [ ] T027 [US3] Implement error handling with graceful fallback when backend services are unavailable in Phase-2 Chatbot using Nextjs/frontend/components/ErrorBoundary.js

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - API Contract Implementation (Priority: P4)

**Goal**: Implement and test all API endpoints according to the contract specification

**Independent Test**: Verify all API endpoints conform to the OpenAPI contract and handle errors properly

### Implementation for User Story 4

- [ ] T028 [P] [US4] Implement /api/chatkit/session POST endpoint per contract in Phase-2 Chatbot using Nextjs/backend/api/sessions.py
- [ ] T029 [P] [US4] Implement /api/rag/query POST endpoint with contextual support per contract in Phase-2 Chatbot using Nextjs/backend/api/rag.py
- [ ] T030 [P] [US4] Implement /api/content/search GET endpoint per contract in Phase-2 Chatbot using Nextjs/backend/api/search.py
- [ ] T031 [US4] Add request/response validation for all endpoints using Pydantic models in Phase-2 Chatbot using Nextjs/backend/schemas/
- [ ] T032 [US4] Implement proper error responses per contract specification in Phase-2 Chatbot using Nextjs/backend/exceptions/

**Checkpoint**: API endpoints conform to contract and are fully tested

---

## Phase 7: User Story 5 - Integration and Testing (Priority: P5)

**Goal**: Integrate all components and perform comprehensive testing

**Independent Test**: End-to-end functionality verification across all components

### Implementation for User Story 5

- [ ] T033 [US5] Integrate Phase-2 Chatbot using Nextjs/frontend with Phase-2 Chatbot using Nextjs/backend API endpoints for complete workflow
- [ ] T034 [US5] Conduct end-to-end testing of RAG functionality with book content from Phase-2 Chatbot using Nextjs/docs/
- [ ] T035 [US5] Perform performance testing for response times under 2 seconds
- [ ] T036 [US5] Test error handling with 3 retry attempts and 10s timeout
- [ ] T037 [US5] Verify data retention with user consent management
- [ ] T038 [US5] Test graceful failure mechanisms when Qdrant is unavailable

**Checkpoint**: Complete system integration and validation

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T039 [P] Documentation updates in Phase-2 Chatbot using Nextjs/docs/
- [ ] T040 Code cleanup and refactoring
- [ ] T041 Performance optimization across all stories
- [ ] T042 [P] Additional unit tests in Phase-2 Chatbot using Nextjs/backend/tests/ and Phase-2 Chatbot using Nextjs/frontend/tests/
- [ ] T043 Security hardening
- [ ] T044 [P] Create Dockerfile for backend deployment in Phase-2 Chatbot using Nextjs/backend/Dockerfile
- [ ] T045 [P] Create docker-compose.yml for local development and testing in Phase-2 Chatbot using Nextjs/docker-compose.yml
- [ ] T046 [P] Configure backend for deployment on Render platform with proper environment variables in Phase-2 Chatbot using Nextjs/backend/render.yaml or Phase-2 Chatbot using Nextjs/backend/.render.yaml
- [ ] T047 Deploy backend to Render with auto-deploy configuration
- [ ] T048 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 (Qdrant indexing completed)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Depends on US2 (API endpoints available)
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - May integrate with US2 but should be independently testable
- **User Story 5 (P5)**: Depends on US2 and US3 (integration story)

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 2

```bash
# Launch all components for User Story 2 together:
Task: "Initialize FastAPI application with OpenAI Agent SDK integration in backend/main.py"
Task: "Implement ChatKit session endpoint (/api/chatkit/session) with OAuth 2.0 and JWT token support in backend/api/sessions.py"
Task: "Create custom Agent Tool that interfaces with Qdrant through qdrant-mcp-server for vector database retrieval in backend/tools/rag_tool.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Data Ingestion Pipeline)
4. **STOP and VALIDATE**: Test Data Ingestion independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Data Ingestion)
   - Developer B: User Story 2 (Backend API)
   - Developer C: User Story 3 (Frontend)
   - Developer D: User Story 4 (API Contract)
3. Stories complete and integrate independently
4. Final developer: Deployment tasks (Docker/Render configuration and deployment)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence