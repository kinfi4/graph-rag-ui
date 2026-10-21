"""Ingestion pipeline: documents -> chunks -> LLM extraction -> validation -> Neo4j -> embeddings.

Used both by the demo-graph build script and by uploads from the web UI.
"""

import io
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from hashlib import sha256

from tqdm import tqdm

from synapse_rag.config import settings
from synapse_rag.extraction import ExtractionCache, Extractor, validate_extraction
from synapse_rag.llm import ChatClient, Embedder, get_chat, get_embedder
from synapse_rag.store import GraphStore


@dataclass
class IngestProgress:
    stage: str  # extracting | embedding | done | error
    done: int = 0
    total: int = 0
    message: str = ""


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    failed_chunks: int = 0
    entities: int = 0
    triples: int = 0
    dropped_entities: int = 0
    dropped_triples: int = 0
    repaired_endpoints: int = 0
    embedded: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def read_upload(filename: str, data: bytes) -> str:
    """Plain text from an uploaded .txt / .md / .pdf file."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing; paragraphs longer than `max_chars` are split by sentences."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        sentence_chunk = ""
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if sentence_chunk and len(sentence_chunk) + len(sentence) + 1 > max_chars:
                pieces.append(sentence_chunk)
                sentence_chunk = sentence
            else:
                sentence_chunk = f"{sentence_chunk} {sentence}".strip()
        if sentence_chunk:
            pieces.append(sentence_chunk)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def canonical_description(name: str, description: str) -> str:
    """Text that gets embedded for an entity (same canonical form as the research code)."""
    parts = sorted({part.strip() for part in description.split(" | ") if part.strip()})
    return f"**{name.title()}**: {' | '.join(parts)}"


class Ingestor:
    def __init__(
        self,
        store: GraphStore,
        chat: ChatClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._chat = chat or get_chat()
        self._embedder = embedder or get_embedder()
        self._extractor = Extractor(
            self._chat, ExtractionCache(settings.cache_dir / "extractions.jsonl")
        )

    def ingest(
        self,
        documents: list[tuple[str, str, str]],
        on_progress: Callable[[IngestProgress], None],
        chunk_chars: int | None = None,
        show_progress: bool = False,
    ) -> IngestReport:
        """Ingest (title, text, source) documents. `chunk_chars=None` keeps each doc whole."""
        started = time.perf_counter()
        report = IngestReport(documents=len(documents))
        chunks = [
            (title, source, chunk)
            for title, text, source in documents
            for chunk in (chunk_text(text, chunk_chars) if chunk_chars else [text])
        ]
        report.chunks = len(chunks)

        iterator = tqdm(chunks, desc="Ingesting") if show_progress else chunks
        for index, (title, source, chunk) in enumerate(iterator):
            on_progress(IngestProgress("extracting", index, len(chunks), title))
            try:
                raw = self._extractor.extract(title, chunk)
            except Exception as exc:  # keep going: one bad chunk must not kill an upload
                report.failed_chunks += 1
                report.errors.append(f"{title}: {exc}")
                continue
            clean, validation = validate_extraction(raw, title)
            report.entities += len(clean.entities)
            report.triples += len(clean.triples)
            report.dropped_entities += len(validation.dropped_entities)
            report.dropped_triples += validation.dropped_triples
            report.repaired_endpoints += len(validation.repaired_endpoints)

            paragraph_id = f"{title}__{sha256(chunk.encode()).hexdigest()[:10]}"
            self._store.write_chunk(title, source, paragraph_id, chunk, clean)

        on_progress(IngestProgress("embedding", 0, 0, "embedding entity descriptions"))
        self._store.update_doc_counts()
        report.embedded = self.embed_pending()

        report.seconds = round(time.perf_counter() - started, 1)
        on_progress(IngestProgress("done", len(chunks), len(chunks), "done"))
        return report

    def embed_pending(self) -> int:
        """Embed entities whose canonical description changed since the last run."""
        pending = []
        for row in self._store.entities_for_embedding():
            text = canonical_description(row["name"], row["description"])
            digest = sha256(text.encode()).hexdigest()
            if digest != row["description_hash"]:
                pending.append({"name": row["name"], "text": text, "description_hash": digest})
        if not pending:
            return 0

        vectors = self._embedder.embed_many([p["text"] for p in pending])
        self._store.set_embeddings(
            [
                {"name": p["name"], "description_hash": p["description_hash"], "embedding": v}
                for p, v in zip(pending, vectors)
            ]
        )
        return len(pending)
