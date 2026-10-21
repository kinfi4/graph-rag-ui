"""Question answering: analyze + route -> retrieve -> generate -> validate -> refine.

    1. Analyzer (structured output): rewrites the message into a standalone English question
       using chat history (memory), picks a route and extracts entities for graph lookup.
    2. Router: graph_qa -> QASA retrieval; overview -> graph statistics; smalltalk -> no tools.
    3. Generator (structured output): reasoning -> answer -> explanation -> evidence ids.
    4. Validator: evidence ids must exist in the context, the answer must be grounded in it.
    5. Refinement: if the answer is missing/unsupported, retrieve again with a wider traversal.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from synapse_rag.llm import ChatClient, Embedder
from synapse_rag.metrics import normalize_answer
from synapse_rag.retrieval import Mode, QueryEntity, RetrievalParams, RetrievalResult, Retriever
from synapse_rag.store import GraphStore

Route = Literal["graph_qa", "overview", "smalltalk"]
Status = Literal["answered", "unverified", "not_found", "info"]

LANGUAGES = {"en": "English", "ru": "Russian", "uk": "Ukrainian"}
MAX_GRAPH_NODES = 70


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(description="Self-contained question in English")
    route: Route
    language: str = Field(description="ISO 639-1 code of the language the user wrote in")
    entities: list[QueryEntity]


class GraphAnswer(BaseModel):
    reasoning: str
    answer: str
    explanation: str
    evidence: list[str]
    sufficient: bool


class OverviewAnswer(BaseModel):
    answer: str
    highlights: list[str]


class Reply(BaseModel):
    answer: str


ANALYZE_PROMPT = """You analyze a user message for a question-answering assistant over a \
knowledge graph built from documents (Wikipedia-style articles about people, films, books, \
places, organizations, events).

Conversation so far (oldest first):
{history}

Current user message: {message}

Fill the JSON fields:
- standalone_question: the current message rewritten as a self-contained question in \
English. Resolve pronouns and references using the conversation ("he", "that film", \
"а где он родился?"). Keep proper names exactly.
- route:
  - "graph_qa": a factual question about specific people, works, places, organizations or \
events that the documents may answer.
  - "overview": a question about the knowledge base as a whole (what it contains, which \
topics, documents or most connected entities).
  - "smalltalk": greetings, thanks, questions about the assistant itself, or requests \
unrelated to the documents (jokes, coding, weather, math).
- language: ISO 639-1 code of the language of the current user message ("en", "ru", "uk"...).
- entities: only for graph_qa. Named entities from the standalone question for graph lookup:
  - name: full proper name in English, lowercase, as a Wikipedia title would be \
(translate or transliterate non-English names)
  - search_hint: the name plus 1-2 disambiguating words from the question \
(e.g. "uhf film", "titanic ship")
  Proper nouns only, never question words or generic words. At least one entity for graph_qa.
"""

ANSWER_PROMPT = """Answer the question using ONLY the knowledge graph below.

Path notation:
- A --[relation]--> B means "A relation B" (einstein --[born in]--> ulm: Einstein was born in Ulm)
- A <--[relation]-- B means "B relation A" (ulm <--[born in]-- einstein: Einstein was born in Ulm)

Knowledge graph paths:
{paths}

Entity descriptions:
{entities}

Question: {question}

Fill the JSON fields:
- reasoning: trace the connections step by step in 2-4 short sentences, citing ids like P3, E5.
- answer: ONLY the final answer, a single entity name or short phrase copied from the graph \
(or "yes"/"no"). Empty string if the graph does not contain the answer.
- explanation: one or two sentences in {language} explaining to the user how the answer \
follows from the graph.
- evidence: ids of the paths/entities that support the answer, e.g. ["P3", "E5"].
- sufficient: true only if the graph above really contains the answer.
"""

OVERVIEW_PROMPT = """You describe a knowledge graph built from documents. Using ONLY the data \
below, answer the user's question in {language}.

Graph size: {stats}
Entity types: {types}
Most connected entities:
{hubs}
Sample documents: {documents}

Question: {question}

Fill the JSON fields:
- answer: 2-4 sentences.
- highlights: 3-6 short bullet points naming notable entities or topics.
"""

SMALLTALK_PROMPT = """You are Synapse, an assistant that answers questions about documents \
loaded into a knowledge graph ({stats}). Reply to the user's message in {language}, in 1-2 \
sentences. For greetings or thanks, answer politely and suggest asking about people, films, \
places or organizations from the documents. For unrelated requests, say that you only answer \
questions about the loaded documents.

User message: {message}
"""


@dataclass
class Context:
    """Numbered context shown to the generator; ids let it cite evidence."""

    path_ids: dict[str, dict] = field(default_factory=dict)
    entity_ids: dict[str, dict] = field(default_factory=dict)
    paths_text: str = "No paths found."
    entities_text: str = "No entities found."

    @property
    def text(self) -> str:
        return f"{self.paths_text}\n{self.entities_text}"


def path_to_text(path: dict) -> str:
    names, chain = path["names"], path["names"][0]
    for i, edge in enumerate(path["edges"]):
        arrow = f" --[{edge['relation']}]--> " if edge["forward"] else f" <--[{edge['relation']}]-- "
        chain += arrow + names[i + 1]
    return chain


def build_context(retrieval: RetrievalResult, max_description: int = 300) -> Context:
    context = Context()
    for i, path in enumerate(retrieval.paths, 1):
        context.path_ids[f"P{i}"] = path
    entities = [e for e in retrieval.entities if e.get("description")]
    for i, entity in enumerate(entities, 1):
        context.entity_ids[f"E{i}"] = entity
    if context.path_ids:
        context.paths_text = "\n".join(
            f"[{pid}] {path_to_text(p)}" for pid, p in context.path_ids.items()
        )
    if context.entity_ids:
        context.entities_text = "\n".join(
            f"[{eid}] {e['name']} ({e.get('type') or 'UNKNOWN'}): "
            f"{e['description'][:max_description]}"
            for eid, e in context.entity_ids.items()
        )
    return context


def validate_evidence(raw_ids: list[str], context: Context) -> list[str]:
    """Keep only ids that exist in the context; a bare number is read as a path id."""
    valid: list[str] = []
    for raw in raw_ids:
        match = re.fullmatch(r"\[?\s*([PE]?)\s*(\d+)\s*\]?", raw.strip().upper())
        if not match:
            continue
        evidence_id = f"{match[1] or 'P'}{match[2]}"
        known = evidence_id in context.path_ids or evidence_id in context.entity_ids
        if known and evidence_id not in valid:
            valid.append(evidence_id)
    return valid


def is_grounded(answer: str, context_text: str, min_overlap: float = 0.6) -> bool:
    """Most answer tokens must appear in the retrieved context (catches answers from memory)."""
    tokens = normalize_answer(answer).split()
    if not tokens:
        return False
    if tokens in (["yes"], ["no"]):
        return True
    context_tokens = set(normalize_answer(context_text).split())
    return sum(token in context_tokens for token in tokens) / len(tokens) >= min_overlap


def _format_history(history: list[dict[str, str]], max_turns: int = 6) -> str:
    lines = [
        f"{turn['role']}: {turn['content'][:300]}"
        for turn in history[-max_turns:]
        if turn.get("content")
    ]
    return "\n".join(lines) if lines else "(empty)"


def _language_name(code: str) -> str:
    return LANGUAGES.get(code.lower()[:2], "the language of the user's message")


class Pipeline:
    def __init__(self, store: GraphStore, chat: ChatClient, embedder: Embedder) -> None:
        self._store = store
        self._chat = chat
        self._retriever = Retriever(store, embedder)

    def analyze(self, message: str, history: list[dict[str, str]]) -> QueryAnalysis:
        return self._chat.structured(
            ANALYZE_PROMPT.format(history=_format_history(history), message=message),
            QueryAnalysis,
            max_tokens=600,
        )

    def ask(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        mode: Mode = "qasa",
        refine: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        timings = {"analyze": 0.0, "retrieve": 0.0, "generate": 0.0}

        analysis = self.analyze(message, history or [])
        timings["analyze"] = time.perf_counter() - started

        if analysis.route == "smalltalk":
            result = self._smalltalk(message, analysis, timings)
        elif analysis.route == "overview":
            result = self._overview(analysis, timings)
        elif not analysis.entities:
            # Router said graph_qa but found nothing to look up: fall back to vector search.
            result = self._graph_qa(analysis, "naive", refine, timings)
        else:
            result = self._graph_qa(analysis, mode, refine, timings)

        timings["total"] = time.perf_counter() - started
        return {
            "question": message,
            "standalone_question": analysis.standalone_question,
            "route": analysis.route,
            "language": analysis.language,
            "query_entities": [e.model_dump() for e in analysis.entities],
            "llm": self._chat.name,
            "timings": {k: round(v, 2) for k, v in timings.items()},
            **result,
        }

    # ---------- routes ----------

    def _graph_qa(
        self, analysis: QueryAnalysis, mode: Mode, refine: bool, timings: dict[str, float]
    ) -> dict[str, Any]:
        params = RetrievalParams()
        attempts: list[dict[str, Any]] = []
        candidates = []

        for _ in range(2 if refine else 1):
            retrieval = self._retriever.retrieve(
                analysis.standalone_question, analysis.entities, mode, params
            )
            timings["retrieve"] += retrieval.seconds
            context = build_context(retrieval)

            t = time.perf_counter()
            answer = self._chat.structured(
                ANSWER_PROMPT.format(
                    paths=context.paths_text,
                    entities=context.entities_text,
                    question=analysis.standalone_question,
                    language=_language_name(analysis.language),
                ),
                GraphAnswer,
                max_tokens=700,
            )
            timings["generate"] += time.perf_counter() - t

            evidence = validate_evidence(answer.evidence, context)
            grounded = is_grounded(answer.answer, context.text)
            ok = answer.sufficient and grounded
            attempts.append(
                {
                    "max_steps": params.max_steps,
                    "threshold": params.threshold,
                    "paths": len(retrieval.paths),
                    "entities": len(retrieval.entities),
                    "answer": answer.answer,
                    "sufficient": answer.sufficient,
                    "grounded": grounded,
                }
            )
            candidates.append((ok, grounded, retrieval, context, answer, evidence))
            if ok:
                break
            params = params.expanded()

        ok, grounded, retrieval, context, answer, evidence = next(
            (c for c in candidates if c[0]),
            next((c for c in candidates if c[1]), candidates[-1]),
        )
        if ok:
            status: Status = "answered"
        elif answer.answer.strip():
            status = "unverified"
        else:
            status = "not_found"

        return {
            "mode": retrieval.mode,
            "status": status,
            "answer": answer.answer,
            "explanation": answer.explanation,
            "reasoning": answer.reasoning,
            "evidence": evidence,
            "grounded": grounded,
            "attempts": attempts,
            "refined": len(attempts) > 1,
            "seeds": retrieval.seeds,
            "paths": [
                {
                    "id": pid,
                    "text": path_to_text(p),
                    "names": p["names"],
                    "weight": round(p["path_weight"], 4),
                }
                for pid, p in context.path_ids.items()
            ],
            "steps": max(0, len(retrieval.trace) - 1),
            "graph": self._graph_payload(retrieval, context, evidence),
            "sources": self._sources(retrieval, context, evidence),
        }

    def _overview(self, analysis: QueryAnalysis, timings: dict[str, float]) -> dict[str, Any]:
        t = time.perf_counter()
        hubs = self._store.hub_entities(20)
        types = self._store.type_counts()
        stats = self._store.stats()
        documents = [d["title"] for d in self._store.documents(15)]
        timings["retrieve"] = time.perf_counter() - t

        t = time.perf_counter()
        overview = self._chat.structured(
            OVERVIEW_PROMPT.format(
                language=_language_name(analysis.language),
                stats=", ".join(f"{v} {k}" for k, v in stats.items()),
                types=", ".join(f"{row['type']}: {row['n']}" for row in types),
                hubs="\n".join(
                    f"- {h['name']} ({h['type']}, {h['degree']} relations): "
                    f"{(h['description'] or '')[:160]}"
                    for h in hubs
                ),
                documents="; ".join(documents),
                question=analysis.standalone_question,
            ),
            OverviewAnswer,
            max_tokens=700,
        )
        timings["generate"] = time.perf_counter() - t

        names = [h["name"] for h in hubs]
        top_degree = max((h["degree"] for h in hubs), default=1) or 1
        details = {d["name"]: d for d in self._store.entity_details(names)}
        nodes = [
            {
                **_node_details(details.get(h["name"], h)),
                "id": h["name"],
                "is_seed": False,
                "evidence": False,
                "activation": [h["degree"] / top_degree],
            }
            for h in hubs
        ]
        return {
            "mode": "overview",
            "status": "info",
            "answer": overview.answer,
            "explanation": "",
            "highlights": overview.highlights,
            "reasoning": "",
            "evidence": [],
            "grounded": True,
            "attempts": [],
            "refined": False,
            "seeds": [],
            "paths": [],
            "steps": 0,
            "graph": {"nodes": nodes, "links": self._links(names, [], [])},
            "sources": [],
        }

    def _smalltalk(
        self, message: str, analysis: QueryAnalysis, timings: dict[str, float]
    ) -> dict[str, Any]:
        t = time.perf_counter()
        stats = self._store.stats()
        reply = self._chat.structured(
            SMALLTALK_PROMPT.format(
                stats=f"{stats['documents']} documents, {stats['entities']} entities",
                language=_language_name(analysis.language),
                message=message,
            ),
            Reply,
            max_tokens=200,
            temperature=0.3,
        )
        timings["generate"] = time.perf_counter() - t
        return {
            "mode": "none",
            "status": "info",
            "answer": reply.answer,
            "explanation": "",
            "reasoning": "",
            "evidence": [],
            "grounded": True,
            "attempts": [],
            "refined": False,
            "seeds": [],
            "paths": [],
            "steps": 0,
            "graph": {"nodes": [], "links": []},
            "sources": [],
        }

    # ---------- explainability payload ----------

    def _graph_payload(
        self, retrieval: RetrievalResult, context: Context, evidence: list[str]
    ) -> dict[str, Any]:
        evidence_paths = [context.path_ids[e]["names"] for e in evidence if e in context.path_ids]
        evidence_names = {n for names in evidence_paths for n in names} | {
            context.entity_ids[e]["name"] for e in evidence if e in context.entity_ids
        }
        all_paths = [p["names"] for p in retrieval.paths]

        ordered: list[str] = []
        for name in [
            *retrieval.seeds,
            *(n for names in evidence_paths for n in names),
            *(n for names in all_paths for n in names),
            *(e["name"] for e in retrieval.entities),
        ]:
            if name not in ordered:
                ordered.append(name)
        names = ordered[:MAX_GRAPH_NODES]

        details = {d["name"]: d for d in self._store.entity_details(names)}
        trace = retrieval.trace or [{}]
        nodes = [
            {
                **_node_details(details.get(name, {"name": name})),
                "id": name,
                "is_seed": name in retrieval.seeds,
                "evidence": name in evidence_names,
                "activation": [round(step.get(name, 0.0), 4) for step in trace],
            }
            for name in names
        ]
        return {"nodes": nodes, "links": self._links(names, all_paths, evidence_paths)}

    def _links(
        self, names: list[str], paths: list[list[str]], evidence_paths: list[list[str]]
    ) -> list[dict[str, Any]]:
        def pairs(path_list: list[list[str]]) -> set[frozenset[str]]:
            return {frozenset(p[i : i + 2]) for p in path_list for i in range(len(p) - 1)}

        on_path, on_evidence = pairs(paths), pairs(evidence_paths)
        return [
            {
                **edge,
                "on_path": frozenset((edge["source"], edge["target"])) in on_path,
                "evidence": frozenset((edge["source"], edge["target"])) in on_evidence,
            }
            for edge in self._store.edges_between(names)
        ]

    def _sources(
        self, retrieval: RetrievalResult, context: Context, evidence: list[str]
    ) -> list[dict[str, Any]]:
        names = [n for e in evidence if e in context.path_ids for n in context.path_ids[e]["names"]]
        names += [context.entity_ids[e]["name"] for e in evidence if e in context.entity_ids]
        names = list(dict.fromkeys(names or retrieval.seeds))
        return self._store.supporting_paragraphs(names) if names else []


def _node_details(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": row.get("type") or "UNKNOWN",
        "description": (row.get("description") or "")[:500],
        "documents": row.get("documents", []),
        "degree": row.get("degree", 0),
    }
