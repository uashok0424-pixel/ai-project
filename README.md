# RAG Document Q&A API
FastAPI service: chunks .txt files in `docs/`, embeds them into ChromaDB, retrieves top-k chunks per question, and (optionally) generates a grounded answer with an LLM. Repeat queries are served from an in-memory LRU cache.

## Run
    pip install -r requirements.txt
    uvicorn app:app --reload
    # optional LLM answers: set ANTHROPIC_API_KEY
Docker: `docker build -t rag-qa . && docker run -p 8000:8000 rag-qa`

## Endpoints
- POST /ask {"question": "...", "top_k": 3}
- POST /ingest  (re-index docs/)
- GET  /health  (chunk count + cache stats)

## Next steps
Evaluation set, reranking, auth, async ingestion queue.
