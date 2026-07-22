# Production REST & Streaming API Server

The API server (`api/main.py`) exposes OpenAI-compatible REST endpoints using FastAPI and Uvicorn.

## Running the Server

```bash
# Start server using Uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints Reference

### 1. Health Check (`GET /health`)
- **Returns**: `{"status": "ok"}`

### 2. Model Information (`GET /v1/models` and `GET /model`)
- **Returns**: Model configuration, parameter count, device, and precision metadata.

### 3. Text Completion (`POST /v1/completions` or `POST /generate`)
- **Request Body**:
  ```json
  {
    "prompt": "The future of AI is",
    "max_tokens": 64,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "stream": false
  }
  ```
- **Response**: Standard OpenAI text completion payload or Server-Sent Events (SSE) stream if `stream: true`.

### 4. Chat Completion (`POST /v1/chat/completions`)
- **Request Body**:
  ```json
  {
    "messages": [
      {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 128
  }
  ```

### 5. Tokenize (`POST /tokenize`)
- **Request Body**: `{"text": "Hello world"}`
- **Response**: `{"ids": [12, 45, ...], "tokens": [...], "num_tokens": 2}`

### 6. Detokenize (`POST /detokenize`)
- **Request Body**: `{"ids": [12, 45]}`
- **Response**: `{"text": "Hello world"}`
