"""RAG Document Q&A API: FastAPI + ChromaDB embeddings + optional LLM answer + query cache."""
import os, glob, hashlib
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb

app = FastAPI(title="RAG Document Q&A API")
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_or_create_collection("docs")  # uses Chroma's default embedding model

def chunk(text, size=500, overlap=80):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size]); i += size - overlap
    return out

def ingest_folder(folder="docs"):
    n = 0
    for path in glob.glob(f"{folder}/*.txt"):
        text = open(path, encoding="utf-8").read()
        for j, c in enumerate(chunk(text)):
            cid = hashlib.md5(f"{path}-{j}".encode()).hexdigest()
            col.upsert(ids=[cid], documents=[c], metadatas=[{"source": os.path.basename(path)}])
            n += 1
    return n

@app.on_event("startup")
def _startup():
    ingest_folder()

class Query(BaseModel):
    question: str
    top_k: int = 3

@lru_cache(maxsize=256)  # cache repeat questions -> lower latency
def _answer(question: str, top_k: int):
    res = col.query(query_texts=[question], n_results=top_k)
    docs, metas = res["documents"][0], res["metadatas"][0]
    sources = [m["source"] for m in metas]
    context = "\n---\n".join(docs)
    answer = None
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:  # optional LLM generation grounded in retrieved context
        import anthropic
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"), max_tokens=400,
            messages=[{"role": "user", "content":
                f"Answer only from the context. If missing, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"}])
        answer = msg.content[0].text
    return {"answer": answer, "retrieved_chunks": docs, "sources": sources}

@app.post("/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "question required")
    return _answer(q.question.strip(), q.top_k)

@app.post("/ingest")
def ingest():
    return {"chunks_indexed": ingest_folder()}

@app.get("/health")
def health():
    return {"status": "ok", "chunks": col.count(), "cache": _answer.cache_info()._asdict()}
