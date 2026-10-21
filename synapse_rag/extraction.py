"""Document -> entities + relations (structured output), then validation and repair.

The prompt is ported from the QASA research code (graph-rag-system/src/graph/extractor.py),
with one extra CONCEPT type so uploaded non-Wikipedia documents also produce a graph.
"""

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field

from synapse_rag.llm import ChatClient

PROMPT_VERSION = "v1"

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "WORK", "CONCEPT"]


class Entity(BaseModel):
    name: str = Field(description="Full proper name in Title Case")
    type: EntityType
    description: str = Field(description="One sentence describing this entity in context")


class Triple(BaseModel):
    subject: str
    relation: str
    object: str


class ExtractionResult(BaseModel):
    entities: list[Entity]
    triples: list[Triple]


def _bounded_schema() -> dict:
    """ExtractionResult schema with length bounds for the LLM call.

    Ollama compiles the schema into the decoding grammar, so these bounds stop a small model
    from looping on a list until the token limit (observed on ~5% of documents).
    """
    schema = ExtractionResult.model_json_schema()
    schema["properties"]["entities"]["maxItems"] = 20
    schema["properties"]["triples"]["maxItems"] = 25
    entity = schema["$defs"]["Entity"]["properties"]
    triple = schema["$defs"]["Triple"]["properties"]
    entity["name"]["maxLength"], entity["description"]["maxLength"] = 80, 180
    for name, limit in (("subject", 80), ("relation", 40), ("object", 80)):
        triple[name]["maxLength"] = limit
    return schema


EXTRACTION_SCHEMA = _bounded_schema()


EXTRACTION_PROMPT = """You are a Knowledge Graph Specialist. Extract named entities and \
relationships from the text.

## Entity Types (ONLY extract these)
- PERSON: Individuals with proper names (e.g., "Albert Einstein", "Marie Curie")
- ORGANIZATION: Companies, institutions, teams, bands (e.g., "NASA", "The Beatles")
- LOCATION: Geographic places with proper names (e.g., "Tokyo", "Mount Everest")
- EVENT: Named events, wars, competitions (e.g., "World War II", "2008 Summer Olympics")
- WORK: Creative works, publications, products (e.g., "Harry Potter", "The New York Times")
- CONCEPT: Named theories, methods, technologies (e.g., "Theory of Relativity", "Transformer")

## DO NOT Extract
- Pronouns or references: he, she, they, it, this, that, the former, the latter
- Generic terms: the company, the author, a person, the team, the city
- Partial dates: '47, the 90s, last year (only named dates like "March 2020")
- Bare numbers: 100, 37 (unless part of a name like "Apollo 11")
- Common nouns without proper names: city, country, book, song, album, film
- Adjectives or descriptors: famous, large, important, American

## Critical Rules
- Use the FULL proper name: "Albert Einstein" not "Einstein", "United States" not "US"
- Use third person: name the subject explicitly, never use pronouns
- Each triple MUST reference entities from the entities list
- Put dates, years and numbers into descriptions and relations, e.g. "born on 22 December 1971"
- Input format is: first line = document title (**title**), next lines = paragraph text
- Consider BOTH the title and the full text when writing descriptions

## Example
Text: "Einstein developed relativity while at the Swiss Patent Office in Bern."
Entities:
- Albert Einstein | PERSON | Physicist who developed the theory of relativity
- Swiss Patent Office | ORGANIZATION | Government office where Einstein worked
- Bern | LOCATION | Swiss city where the Patent Office is located

Triples:
- (Albert Einstein, worked at, Swiss Patent Office)
- (Swiss Patent Office, located in, Bern)

Text:
{text}"""

# Validation: names that slip through the prompt rules on small models.
_BAD_NAMES = {
    "he", "she", "they", "it", "this", "that", "him", "her", "them", "his", "its", "their",
    "the former", "the latter", "the film", "the company", "the author", "the city",
    "the band", "the album", "the song", "the book", "film", "album", "song", "book",
    "city", "country", "person", "unknown", "none", "n/a",
}
_MAX_NAME_LEN = 80


def normalize_name(text: str) -> str:
    """Graph key for an entity: lowercase, single spaces, no wrapping quotes/punctuation."""
    text = " ".join(text.strip().lower().split())
    return text.strip(" \"'`.,;:")


def _is_valid_name(name: str) -> bool:
    return (
        bool(name)
        and name not in _BAD_NAMES
        and len(name) <= _MAX_NAME_LEN
        and not re.fullmatch(r"[\d\s.,:/-]+", name)
    )


@dataclass
class ValidationReport:
    entities_in: int = 0
    triples_in: int = 0
    dropped_entities: list[str] = field(default_factory=list)
    dropped_triples: int = 0
    repaired_endpoints: list[str] = field(default_factory=list)


def validate_extraction(
    raw: ExtractionResult, title: str
) -> tuple[ExtractionResult, ValidationReport]:
    """Normalize names, drop junk entities and broken triples, repair missing endpoints.

    A triple whose endpoint was never listed as an entity gets that endpoint added with a
    synthesized description, so it can still be embedded and take part in traversal.
    """
    report = ValidationReport(entities_in=len(raw.entities), triples_in=len(raw.triples))

    entities: dict[str, Entity] = {}
    for entity in raw.entities:
        name = normalize_name(entity.name)
        if not _is_valid_name(name):
            report.dropped_entities.append(entity.name)
            continue
        description = entity.description.strip()
        if name in entities:
            known = entities[name]
            if description and description not in known.description:
                known.description = f"{known.description} | {description}"
            continue
        entities[name] = Entity(name=name, type=entity.type, description=description)

    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in raw.triples:
        subject, obj = normalize_name(triple.subject), normalize_name(triple.object)
        relation = " ".join(triple.relation.strip().lower().split())
        key = (subject, relation, obj)
        if (
            not relation
            or subject == obj
            or not _is_valid_name(subject)
            or not _is_valid_name(obj)
            or key in seen
        ):
            report.dropped_triples += 1
            continue
        for endpoint in (subject, obj):
            if endpoint not in entities:
                entities[endpoint] = Entity(
                    name=endpoint,
                    type="CONCEPT",
                    description=f"Mentioned in '{title}': {subject} {relation} {obj}.",
                )
                report.repaired_endpoints.append(endpoint)
        seen.add(key)
        triples.append(Triple(subject=subject, relation=relation, object=obj))

    return ExtractionResult(entities=list(entities.values()), triples=triples), report


class ExtractionCache:
    """Append-only JSONL cache keyed by (model, prompt version, text): reruns are free."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._items: dict[str, ExtractionResult] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                try:
                    row = json.loads(line)
                    self._items[row["key"]] = ExtractionResult.model_validate(row["result"])
                except ValueError:  # blank or half-written line (process killed mid-write)
                    continue

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def key(model: str, text: str) -> str:
        return sha256(f"{model}\n{PROMPT_VERSION}\n{text}".encode()).hexdigest()

    def get(self, key: str) -> ExtractionResult | None:
        return self._items.get(key)

    def put(self, key: str, result: ExtractionResult) -> None:
        with self._lock:
            self._items[key] = result
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                row = {"key": key, "result": result.model_dump()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Extractor:
    def __init__(self, chat: ChatClient, cache: ExtractionCache) -> None:
        self._chat = chat
        self._cache = cache

    def cached(self, title: str, text: str) -> ExtractionResult | None:
        return self._cache.get(ExtractionCache.key(self._chat.name, f"**{title}**\n{text}"))

    def extract(self, title: str, text: str) -> ExtractionResult:
        """Raw (unvalidated) extraction for one document chunk, served from cache if possible."""
        doc_text = f"**{title}**\n{text}"
        key = ExtractionCache.key(self._chat.name, doc_text)
        if (cached := self._cache.get(key)) is not None:
            return cached
        result = self._chat.structured(
            EXTRACTION_PROMPT.format(text=doc_text),
            ExtractionResult,
            schema=EXTRACTION_SCHEMA,
            temperature=0.1,
            max_tokens=2000,
            # a truncated (looping) extraction rarely recovers on retry: skip it instead
            retries=0,
        )
        self._cache.put(key, result)
        return result
