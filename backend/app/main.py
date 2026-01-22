from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings

load_dotenv()

app = FastAPI()

# -----------------------------
# Globals / config
# -----------------------------
DATA_DIR = os.getenv("DATA_DIR", "app/data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
COLLECTION_NAME = "documents"
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

# IMPORTANT: set cosine space so distances behave as expected
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------
def extract_pages_from_pdf(path: str):
    """Returns: [{"page": 1, "text": "..."}, ...]"""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages):
    """
    Chunk each page separately so we can cite page numbers.
    Each chunk: {"chunk_index", "page", "text"}
    """
    assert CHUNK_OVERLAP < CHUNK_SIZE

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"] or ""
        page_number = page["page"]

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_str = text[start:end].strip()

            if chunk_str:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "page": page_number,
                        "text": chunk_str,
                    }
                )
                chunk_index += 1

            start = end - CHUNK_OVERLAP

    return chunks


def index_doc_into_chroma(doc_id: str, path: str):
    """
    Extract -> chunk -> embed -> store in Chroma (persistent).
    """
    pages = extract_pages_from_pdf(path)
    full_text = "\n".join(p["text"] for p in pages)

    if not full_text.strip():
        return {"error": "No extractable text found (scanned PDF). OCR not supported in v1."}

    chunks = chunk_pages(pages)
    if not chunks:
        return {"error": "No valid text chunks found."}

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(f"{doc_id}:{c['chunk_index']}")
        documents.append(c["text"])
        metadatas.append(
            {
                "doc_id": doc_id,
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            }
        )

    # Remove any prior entries for doc_id (re-index safety)
    collection.delete(where={"doc_id": doc_id})

    # Compute embeddings ONCE and persist them
    embeddings = embedding_model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return {"pages": len(pages), "chunks": len(chunks)}


def build_prompt(question: str, retrieved_chunks):
    """Put context in prompt. Forbid citations inside answer."""
    context_blocks = []
    for r in retrieved_chunks:
        context_blocks.append(f"(page {r['page']}) {r['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "You are a careful, precise assistant.\n"
        "Answer the user's question using ONLY the context below.\n"
        "Prefer either a short paragraph or a simple list where each item is on its own line.\n"
        "Write in plain text only: do NOT use Markdown formatting such as **bold**.\n"
        "Do NOT include citations like [chunk X] or (page X) inside the answer.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided document.\".\n\n"
        "When listing multiple items, put each item on its own line starting with '- '. Avoid cramming many items into one long sentence.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
    )
    return prompt


def retrieve_top_k(doc_id: str, question: str, top_k: int):
    """Query Chroma directly (cosine space) and return scored chunks.

    Returns list of dicts: {page, text, chunk_index, similarity} sorted by similarity desc.
    """
    q_emb = embedding_model.encode([question]).tolist()

    # Ask Chroma for a bit more than top_k so we can dedupe while keeping diversity.
    n_results = max(top_k * 3, top_k)

    res = collection.query(
        query_embeddings=q_emb,
        n_results=n_results,
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    retrieved = []
    for d, m, dist in zip(docs, metas, dists):
        if not d:
            continue

        sim = None
        if dist is not None:
            # collection is configured with hnsw:space = "cosine" so distance ~= 1 - cosine_similarity
            sim = 1.0 - float(dist)

        retrieved.append(
            {
                "text": d,
                "page": (m or {}).get("page"),
                "chunk_index": (m or {}).get("chunk_index"),
                "similarity": sim,
            }
        )

    # Basic dedupe: keep first occurrence per (page, chunk_index) and then cap to top_k.
    seen_keys = set()
    deduped = []
    for r in retrieved:
        key = (r.get("page"), r.get("chunk_index"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(r)

    deduped.sort(key=lambda x: (x.get("similarity") or 0.0), reverse=True)
    return deduped[:top_k]


def trim_excerpt(text: str, max_chars: int = 260):
    """Cleaner evidence snippet: single paragraph, trimmed at a sentence boundary when possible."""
    if not text:
        return ""

    # Collapse whitespace and bullets into a single flowing paragraph.
    t = " ".join(text.replace("•", " ").split())
    if len(t) <= max_chars:
        return t

    snippet = t[: max_chars + 40]  # small buffer to search for a sentence end
    cut = snippet.rfind(".")
    if cut == -1 or cut < int(max_chars * 0.6):
        # Fallback: hard cut near max_chars
        return t[:max_chars].rstrip() + "…"
    return snippet[: cut + 1].rstrip() + "…"


# -----------------------------
# Schemas
# -----------------------------
class AskRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 5


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload-and-index")
async def upload_and_index(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={"error": "Only PDF files are supported (expected a .pdf file)."},
        )

    doc_id = str(uuid.uuid4())
    content = await file.read()

    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "File is too large. Maximum supported size is 25 MB."},
        )

    saved_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    try:
        with open(saved_path, "wb") as f:
            f.write(content)
    except Exception as exc:  # disk / permissions issues
        raise HTTPException(status_code=500, detail={"error": f"Failed to save file: {exc}"})

    summary = index_doc_into_chroma(doc_id, saved_path)
    if "error" in summary:
        # Keep the PDF on disk so the user can re-index later if needed.
        raise HTTPException(status_code=400, detail=summary)

    return {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "pages": summary["pages"],
        "chunks": summary["chunks"],
        "indexed": True,
    }


@app.post("/ask")
def ask(payload: AskRequest):
    retrieved = retrieve_top_k(payload.doc_id, payload.question, payload.top_k)

    if not retrieved:
        return {
            "doc_id": payload.doc_id,
            "question": payload.question,
            "answer": "I don't know based on the provided document.",
            "sources": [],
            "evidence": [],
        }

    prompt = build_prompt(payload.question, retrieved)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            timeout=20,  # seconds
        )
    except Exception as exc:
        # Surface a clean, standardized error up to the client.
        raise HTTPException(
            status_code=502,
            detail={"error": f"Upstream model error or timeout: {exc}"},
        )
    answer = response.output_text.strip()
    # Light post-processing so the UI stays clean and readable.
    answer = answer.replace("**", "")
    # If the model created inline bullets like "- A - B - C", break them onto new lines.
    if " - " in answer and "\n- " not in answer:
        answer = answer.replace(" - ", "\n- ")

    pages_used = sorted(set(r["page"] for r in retrieved if r.get("page") is not None))

    evidence = []
    # Show up to two of the top retrieved chunks as human-readable evidence snippets.
    for r in retrieved[:2]:
        evidence.append(
            {
                "page": r.get("page"),
                "excerpt": trim_excerpt(r.get("text", ""), 260),
            }
        )

    return {
        "doc_id": payload.doc_id,
        "question": payload.question,
        "answer": answer,
        "sources": pages_used,
        "evidence": evidence,
    }


@app.delete("/docs/{doc_id}")
def delete_doc(doc_id: str):
    """Delete a document's vectors from Chroma and its uploaded PDF.

    Safe to call multiple times; returns success even if parts were already removed.
    """

    # Remove all vectors for this document
    collection.delete(where={"doc_id": doc_id})

    # Remove the uploaded PDF if it exists
    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"Failed to delete PDF: {exc}"})

    return {"ok": True, "doc_id": doc_id}
