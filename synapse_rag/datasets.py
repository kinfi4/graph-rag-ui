"""2WikiMultiHopQA subset used for the demo graph (same HF dataset as the QASA research)."""

import json
from pathlib import Path

import httpx

DATASET = "framolfese/2WikiMultihopQA"
_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def fetch_2wiki(n_questions: int, offset: int = 0) -> list[dict]:
    """Download `n_questions` validation examples, converted to {question, answer, paragraphs}."""
    examples: list[dict] = []
    with httpx.Client(timeout=60) as http:
        while len(examples) < n_questions:
            length = min(100, n_questions - len(examples))
            response = http.get(
                _ROWS_URL,
                params={
                    "dataset": DATASET,
                    "config": "default",
                    "split": "validation",
                    "offset": offset,
                    "length": length,
                },
            )
            response.raise_for_status()
            rows = response.json()["rows"]
            if not rows:
                break
            examples.extend(_convert(row["row"]) for row in rows)
            offset += len(rows)
    return examples


def _convert(row: dict) -> dict:
    supporting = set(row["supporting_facts"]["title"])
    paragraphs = []
    for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        text = " ".join(s.strip() for s in sentences if s and s.strip())
        if text:
            paragraphs.append({"title": title, "text": text, "is_supporting": title in supporting})
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "evidences": row["evidences"],
        "paragraphs": paragraphs,
    }


def unique_documents(examples: list[dict]) -> dict[str, str]:
    """title -> merged text of all unique paragraphs with that title."""
    texts: dict[str, list[str]] = {}
    for example in examples:
        for paragraph in example["paragraphs"]:
            bucket = texts.setdefault(paragraph["title"], [])
            if paragraph["text"] not in bucket:
                bucket.append(paragraph["text"])
    return {title: "\n".join(parts) for title, parts in texts.items()}


def save_examples(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")


def load_examples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
