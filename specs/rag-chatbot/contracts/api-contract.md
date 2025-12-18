# OpenAPI Contract: RAG Chatbot API

## /api/chatkit/session

### POST
**Summary**: Create new chat session

**Request**:
```json
{}
```

**Response**:
```json
{
  "session_id": "string",
  "client_secret": "string",
  "expires_at": "datetime"
}
```

**Errors**:
- 500: ChatKit service unavailable

## /api/rag/query

### POST
**Summary**: RAG query with contextual support

**Request**:
```json
{
  "message": "string",
  "selected_text": "string | null",
  "session_id": "string"
}
```

**Response**:
```json
{
  "response": "string",
  "sources": [
    {
      "id": "string",
      "section": "string",
      "content": "string",
      "relevance_score": "float"
    }
  ],
  "session_id": "string"
}
```

**Errors**:
- 400: Invalid request format
- 404: Session not found
- 500: RAG service error

## /api/content/all

### GET
**Summary**: Retrieve all book content for display

**Response**:
```json
{
  "content": [
    {
      "id": "string",
      "title": "string",
      "section": "string",
      "content": "string",
      "display_order": "integer"
    }
  ],
  "total": "integer"
}
```

**Errors**:
- 500: Content service error

## /api/content/search

### GET
**Summary**: Search book content

**Parameters**:
- query: string (search query)
- limit: integer (max results, default 10)

**Response**:
```json
{
  "results": [
    {
      "id": "string",
      "title": "string",
      "section": "string",
      "content": "string",
      "relevance_score": "float"
    }
  ],
  "total": "integer"
}
```

**Errors**:
- 400: Invalid parameters
- 500: Search service error