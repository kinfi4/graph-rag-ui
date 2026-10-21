"""FastAPI backend: ask questions, upload documents, inspect the graph.

    uv run uvicorn synapse_rag.api:app --port 8000
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from synapse_rag.config import ROOT, settings
from synapse_rag.datasets import load_examples
from synapse_rag.ingest import Ingestor, IngestProgress, read_upload
from synapse_rag.llm import LLMError, get_chat, get_embedder
from synapse_rag.pipeline import Pipeline
from synapse_rag.retrieval import Mode
from synapse_rag.store import GraphStore

app = FastAPI(title="Synapse GraphRAG")

# Uploads are processed one at a time in the background; progress lives in memory.
_jobs: dict[str, dict[str, Any]] = {}
_ingest_pool = ThreadPoolExecutor(max_workers=1)


@lru_cache
def get_store() -> GraphStore:
    store = GraphStore.from_settings()
    store.ensure_schema(settings.embedding_dim)
    return store


@lru_cache
def get_pipeline() -> Pipeline:
    return Pipeline(get_store(), get_chat(), get_embedder())


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = []
    mode: Mode = "qasa"
    refine: bool = True


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        ollama_ok = httpx.get(f"{settings.ollama_url}/api/version", timeout=3).is_success
    except httpx.HTTPError:
        ollama_ok = False
    try:
        neo4j_ok = get_store().ping()
    except Exception:
        neo4j_ok = False
    return {
        "neo4j": neo4j_ok,
        "ollama": ollama_ok,
        "llm": settings.llm_name,
        "embedding": f"ollama/{settings.embedding_model}",
    }


@app.get("/api/stats")
def stats() -> dict[str, int]:
    return _db_call(lambda: get_store().stats())


@app.get("/api/documents")
def documents(limit: int = 200) -> list[dict[str, Any]]:
    return _db_call(lambda: get_store().documents(limit))


@app.get("/api/examples")
def examples() -> list[dict[str, str]]:
    """Demo questions: a few 2Wiki questions with gold answers + router/memory showcases."""
    curated = settings.data_dir / "demo_examples.json"
    if curated.exists():
        return load_examples(curated)

    path = settings.data_dir / "2wiki_demo.json"
    picked = []
    if path.exists():
        seen_types: dict[str, int] = {}
        for example in load_examples(path):
            if seen_types.get(example["type"], 0) < 2:
                seen_types[example["type"]] = seen_types.get(example["type"], 0) + 1
                picked.append({"question": example["question"], "answer": example["answer"]})
    return picked[:8] + [
        {"question": "Что вообще есть в этой базе знаний?", "answer": ""},
        {"question": "Привет! Что ты умеешь?", "answer": ""},
    ]


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    history = [turn.model_dump() for turn in request.history]
    try:
        return get_pipeline().ask(request.question, history, request.mode, request.refine)
    except LLMError as exc:
        raise HTTPException(502, f"LLM error: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Model server unreachable: {exc}") from exc


@app.post("/api/documents")
async def upload(files: list[UploadFile]) -> dict[str, str]:
    documents = []
    for file in files:
        name = file.filename or "upload.txt"
        text = read_upload(name, await file.read())
        if text.strip():
            title = name.rsplit(".", 1)[0]
            documents.append((title, text, "upload"))
    if not documents:
        raise HTTPException(400, "No readable text in the uploaded files")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"id": job_id, "stage": "queued", "done": 0, "total": 0, "message": ""}
    _ingest_pool.submit(_run_ingest, job_id, documents)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(404, "Unknown job")
    return _jobs[job_id]


def _run_ingest(job_id: str, documents: list[tuple[str, str, str]]) -> None:
    def on_progress(progress: IngestProgress) -> None:
        _jobs[job_id].update(
            stage=progress.stage,
            done=progress.done,
            total=progress.total,
            message=progress.message,
        )

    try:
        report = Ingestor(get_store()).ingest(documents, on_progress, chunk_chars=2000)
        _jobs[job_id].update(stage="done", report=report.as_dict())
    except Exception as exc:
        _jobs[job_id].update(stage="error", message=str(exc))


def _db_call(fn):
    try:
        return fn()
    except Exception as exc:
        raise HTTPException(503, f"Neo4j unavailable: {exc}") from exc


# Built frontend (cd frontend && npm run build) is served from the same origin.
_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = _dist / path
        return FileResponse(file if path and file.is_file() else _dist / "index.html")
