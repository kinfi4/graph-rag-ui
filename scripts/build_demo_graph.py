"""Build the demo knowledge graph from a 2WikiMultiHopQA subset.

    uv run python scripts/build_demo_graph.py extract --questions 60   # LLM, no Neo4j needed
    uv run python scripts/build_demo_graph.py load                     # cached extractions -> Neo4j
    uv run python scripts/build_demo_graph.py all --questions 60

Extraction results are cached in data/cache/extractions.jsonl, so both steps are resumable
and re-running with more questions only extracts the new documents.
"""

import argparse
import time

from tqdm import tqdm

from synapse_rag.config import settings
from synapse_rag.datasets import fetch_2wiki, load_examples, save_examples, unique_documents
from synapse_rag.extraction import ExtractionCache, Extractor
from synapse_rag.llm import get_chat

DATASET_PATH = settings.data_dir / "2wiki_demo.json"


def get_examples(n_questions: int | None) -> list[dict]:
    cached = load_examples(DATASET_PATH) if DATASET_PATH.exists() else []
    if n_questions is None or len(cached) >= n_questions:
        return cached[:n_questions] if n_questions else cached
    print(f"Downloading {n_questions} 2Wiki validation questions...")
    examples = fetch_2wiki(n_questions)
    save_examples(examples, DATASET_PATH)
    return examples


def extract(n_questions: int) -> None:
    documents = unique_documents(get_examples(n_questions))
    cache = ExtractionCache(settings.cache_dir / "extractions.jsonl")
    extractor = Extractor(get_chat(), cache)
    print(f"{len(documents)} documents, {len(cache)} cached extractions, LLM={get_chat().name}")

    started, failures = time.perf_counter(), 0
    for title, text in tqdm(documents.items(), desc="Extracting"):
        try:
            extractor.extract(title, text)
        except Exception as exc:  # one broken document must not stop a 1h run
            failures += 1
            tqdm.write(f"[skip] {title}: {exc}")
    print(f"Done in {time.perf_counter() - started:.0f}s, {failures} failures")


def load(n_questions: int | None, cached_only: bool) -> None:
    from synapse_rag.ingest import Ingestor
    from synapse_rag.store import GraphStore

    documents = unique_documents(get_examples(n_questions))
    if cached_only:
        extractor = Extractor(get_chat(), ExtractionCache(settings.cache_dir / "extractions.jsonl"))
        documents = {t: x for t, x in documents.items() if extractor.cached(t, x) is not None}
        print(f"--cached-only: loading {len(documents)} already extracted documents")

    store = GraphStore.from_settings()
    store.ensure_schema(settings.embedding_dim)
    ingestor = Ingestor(store)
    report = ingestor.ingest(
        [(title, text, "2wiki") for title, text in documents.items()],
        on_progress=lambda p: None,
        show_progress=True,
    )
    print(report)
    print(store.stats())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["extract", "load", "all"])
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument(
        "--cached-only", action="store_true", help="load only documents already extracted"
    )
    args = parser.parse_args()

    if args.step in ("extract", "all"):
        extract(args.questions or 60)
    if args.step in ("load", "all"):
        load(args.questions, args.cached_only)


if __name__ == "__main__":
    main()
