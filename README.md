# AI Document Q&A

AI Document Q&A is a small full-stack app for asking questions about PDF documents.

Upload a PDF, wait for it to be indexed, then ask questions in the chat UI. The app uses a simple RAG pipeline so answers are based on retrieved text from the uploaded document, and the UI can show the page-level evidence used for an answer.

## Tech stack

- **Backend:** FastAPI
- **Frontend:** React + Vite
- **Vector storage:** ChromaDB
- **PDF parsing:** pypdf
- **AI:** OpenAI embeddings and chat completions

## Features

- **PDF upload and indexing:** Extracts text from PDFs, chunks it, embeds it, and stores it in ChromaDB.
- **Document Q&A:** Retrieves relevant chunks and asks OpenAI to answer using only that context.
- **Evidence display:** Returns source page numbers and short excerpts so answers are easier to verify.
- **Conservative answers:** If the document does not contain the answer, the app is instructed to say it does not know based on the provided document.
- **Local persistent storage:** Uploaded PDFs and ChromaDB files are stored under `backend/app/data`.

## Project structure

```text
ai-document-qa/
  backend/
    app/
      main.py
    requirements.txt
    start.sh

  frontend/
    src/
      api.js
    package.json
    vite.config.js

  README.md
```

## Requirements

- Python 3.10+
- Node.js LTS
- An OpenAI API key with available quota

## Backend setup

From the `backend` directory:

```bash
pip install -r requirements.txt
```

Create `backend/.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
DATA_DIR=app/data
EMBEDDING_MODEL=text-embedding-3-small
```

Run the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Or use:

```bash
./start.sh
```

The API runs at:

```text
http://localhost:8001
```

Useful endpoints:

- `GET /health`
- `POST /upload-and-index`
- `POST /ask`
- `DELETE /docs/{doc_id}`

## Frontend setup

From the `frontend` directory:

```bash
npm install
npm run dev
```

Vite will usually start at:

```text
http://localhost:5173
```

For local development, `frontend/vite.config.js` proxies API requests to the backend on port `8001`, so the frontend can call API paths like `/upload-and-index`.

## Environment variables

### Backend

```env
OPENAI_API_KEY=your_openai_api_key_here
DATA_DIR=app/data
EMBEDDING_MODEL=text-embedding-3-small
```

### Frontend

For production deployments, set:

```env
VITE_API_BASE_URL=https://your-deployed-backend-url
```

If `VITE_API_BASE_URL` is not set, the frontend uses same-origin API paths. That works locally with the Vite proxy, but production deployments usually need this variable unless the frontend and backend are served from the same domain.

## Where data is stored

By default, backend data is stored locally at:

```text
backend/app/data/
```

That includes:

```text
backend/app/data/uploads/
backend/app/data/chroma/
```

Uploaded PDFs are stored in `uploads`, and ChromaDB stores embeddings/vector data in `chroma`.

On platforms like Railway, Render, or other container hosts, local disk may not be permanent unless persistent volumes are configured.

## Common issues

### `Failed to fetch`

The frontend cannot reach the backend. Check that:

- The backend is running.
- `VITE_API_BASE_URL` points to the deployed backend in production.
- The backend URL uses HTTPS when the frontend is served over HTTPS.
- CORS is enabled on the backend.

### `insufficient_quota`

OpenAI rejected the request because the API key's project has no available credits or quota. Add credits, enable billing, or use an API key from a project with quota.

### `No extractable text found`

The PDF is likely scanned or image-only. This app does not currently perform OCR.

### `The uploaded file could not be read as a valid PDF`

The uploaded file is not a valid PDF or is corrupted.
