"""QASA retrieval: query-aware spreading activation as ONE stateless Cypher query.

Ported from graph-rag-system/src/graph/kin_rag/stateless_cypher_retriever.py. Changes:
- Neo4j's built-in `vector.similarity.cosine` replaces the GDS plugin (it returns
  (1 + cos) / 2, so it is mapped back to cos to keep the original weighting);
- the activation map after every step is returned as `trace` for the UI replay.

Activation never touches the stored graph (state lives in a Cypher map), so concurrent
queries are safe.
"""

import time
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import BaseModel, Field

from synapse_rag.extraction import normalize_name
from synapse_rag.llm import Embedder
from synapse_rag.store import ENTITY_INDEX, GraphStore

Mode = Literal["qasa", "uniform", "naive"]


class QueryEntity(BaseModel):
    name: str = Field(description="Entity name in English as it would appear in the graph")
    search_hint: str = Field(description="Name plus 1-2 disambiguating words for vector search")


@dataclass(frozen=True)
class RetrievalParams:
    max_steps: int = 3
    decay: float = 0.7
    threshold: float = 0.01
    top_k_paths: int = 30
    top_k_entities: int = 30
    seed_k: int = 5

    def expanded(self) -> "RetrievalParams":
        """Wider search for the refinement loop: one more hop, lower cut-off, more context."""
        return replace(
            self,
            max_steps=self.max_steps + 1,
            threshold=self.threshold / 2,
            top_k_paths=self.top_k_paths + 15,
            top_k_entities=self.top_k_entities + 15,
            seed_k=self.seed_k * 2,
        )


@dataclass
class RetrievalResult:
    mode: Mode
    params: RetrievalParams
    seeds: list[str] = field(default_factory=list)
    paths: list[dict] = field(default_factory=list)  # {names, edges, path_weight}
    entities: list[dict] = field(default_factory=list)  # {name, resource, type, description}
    trace: list[dict[str, float]] = field(default_factory=list)  # step -> {entity: resource}
    seconds: float = 0.0


_SEEDS = f"""
UNWIND $entity_data AS ent
OPTIONAL MATCH (exact:Entity {{name: ent.name}})
CALL (ent, exact) {{
    UNWIND CASE WHEN exact IS NOT NULL THEN [exact] ELSE [] END AS e
    RETURN e.name AS seed_name, 1.0 AS seed_score
  UNION ALL
    UNWIND CASE WHEN exact IS NULL THEN [ent] ELSE [] END AS e
    CALL db.index.vector.queryNodes('{ENTITY_INDEX}', $k_per_entity, e.emb)
    YIELD node, score
    RETURN node.name AS seed_name, score AS seed_score
}}
WITH seed_name, max(seed_score) AS seed_score
WITH collect({{name: seed_name, score: seed_score}}) AS raw, max(seed_score) AS mx
WITH [s IN raw | s.name] AS seed_names,
     apoc.map.fromPairs([s IN raw | [s.name, s.score / mx]]) AS resource_map
WITH seed_names, resource_map, [resource_map] AS trace
"""

_STEP_QUERY_AWARE = """
CALL (resource_map) {
    UNWIND keys(resource_map) AS src_name
    WITH src_name, resource_map[src_name] AS src_res, resource_map
    WHERE src_res > $threshold
    MATCH (src:Entity {name: src_name})-[:RELATES_TO]-(dst:Entity)
    WHERE dst.description_embedding IS NOT NULL
      AND (resource_map[dst.name] IS NULL OR resource_map[dst.name] < src_res)
    WITH dst, src_res,
         2 * vector.similarity.cosine(dst.description_embedding, $qe) - 1 AS sim
    WITH dst.name AS dst_name,
         sum(src_res * $decay * CASE WHEN sim > 0 THEN sim ELSE 0 END) AS incoming
    WHERE incoming > $threshold
    RETURN collect({name: dst_name, incoming: incoming}) AS deltas
}
WITH seed_names, trace,
     apoc.map.merge(resource_map, apoc.map.fromPairs(
         [d IN deltas | [d.name, coalesce(resource_map[d.name], 0) + d.incoming]]
     )) AS resource_map
WITH seed_names, resource_map, trace + [resource_map] AS trace
"""

_STEP_UNIFORM = """
CALL (resource_map) {
    UNWIND keys(resource_map) AS src_name
    WITH src_name, resource_map[src_name] AS src_res, resource_map
    WHERE src_res > $threshold
    MATCH (src:Entity {name: src_name})-[:RELATES_TO]-(dst:Entity)
    WHERE resource_map[dst.name] IS NULL OR resource_map[dst.name] < src_res
    WITH dst.name AS dst_name, sum(src_res * $decay) AS incoming
    WHERE incoming > $threshold
    RETURN collect({name: dst_name, incoming: incoming}) AS deltas
}
WITH seed_names, trace,
     apoc.map.merge(resource_map, apoc.map.fromPairs(
         [d IN deltas | [d.name, coalesce(resource_map[d.name], 0) + d.incoming]]
     )) AS resource_map
WITH seed_names, resource_map, trace + [resource_map] AS trace
"""

_COLLECT = """
CALL (seed_names, resource_map) {
    MATCH path = (seed:Entity)-[:RELATES_TO*1..4]-(target:Entity)
    WHERE seed.name IN seed_names
      AND resource_map[target.name] IS NOT NULL
      AND seed <> target
      AND ALL(n IN nodes(path) WHERE resource_map[n.name] IS NOT NULL)
      AND ALL(i IN range(0, size(nodes(path)) - 2)
          WHERE NOT nodes(path)[i] IN nodes(path)[(i + 1)..])
    WITH reduce(w = 0.0, n IN nodes(path) | w + coalesce(resource_map[n.name], 0))
             / size(nodes(path)) AS path_weight,
         [n IN nodes(path) | n.name] AS names,
         [i IN range(0, size(relationships(path)) - 1) |
             {relation: relationships(path)[i].relation,
              forward: startNode(relationships(path)[i]) = nodes(path)[i]}] AS edges
    ORDER BY path_weight DESC
    LIMIT $top_k_paths
    RETURN collect({names: names, edges: edges, path_weight: path_weight}) AS paths
}
CALL (resource_map) {
    UNWIND keys(resource_map) AS name
    WITH name, resource_map[name] AS resource
    ORDER BY resource DESC LIMIT $top_k_entities
    MATCH (e:Entity {name: name})
    RETURN collect({name: e.name, resource: resource,
                    type: e.type, description: e.description}) AS entities
}
RETURN seed_names, paths, entities, trace
"""

_NAIVE = f"""
CALL db.index.vector.queryNodes('{ENTITY_INDEX}', $top_k_entities, $qe)
YIELD node, score
RETURN collect({{name: node.name, resource: score,
                type: node.type, description: node.description}}) AS entities
"""


class Retriever:
    def __init__(self, store: GraphStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(
        self,
        question: str,
        query_entities: list[QueryEntity],
        mode: Mode = "qasa",
        params: RetrievalParams = RetrievalParams(),
    ) -> RetrievalResult:
        started = time.perf_counter()
        result = RetrievalResult(mode=mode, params=params)
        query_embedding = self._embedder.embed(question)

        if mode == "naive":
            rows = self._store.run(
                _NAIVE, qe=query_embedding, top_k_entities=params.top_k_entities
            )
            result.entities = rows[0]["entities"] if rows else []
            result.trace = [{e["name"]: e["resource"] for e in result.entities}]
        elif query_entities:
            hints = self._embedder.embed_many([e.search_hint for e in query_entities])
            entity_data = [
                {"name": normalize_name(e.name), "emb": emb}
                for e, emb in zip(query_entities, hints)
            ]
            step = _STEP_QUERY_AWARE if mode == "qasa" else _STEP_UNIFORM
            rows = self._store.run(
                _SEEDS + step * params.max_steps + _COLLECT,
                entity_data=entity_data,
                k_per_entity=max(1, params.seed_k // len(entity_data)),
                qe=query_embedding,
                decay=params.decay,
                threshold=params.threshold,
                top_k_paths=params.top_k_paths,
                top_k_entities=params.top_k_entities,
            )
            if rows:
                row = rows[0]
                result.seeds = row["seed_names"]
                result.paths = deduplicate_paths(row["paths"])
                result.entities = row["entities"]
                result.trace = row["trace"]

        result.seconds = time.perf_counter() - started
        return result


def deduplicate_paths(rows: list[dict]) -> list[dict]:
    """Drop reversed duplicates and paths that are a prefix of a longer kept path."""
    seen: set[tuple[str, ...]] = set()
    candidates: list[dict] = []
    for row in rows:
        key = tuple(row["names"])
        if key in seen or tuple(reversed(key)) in seen:
            continue
        seen.add(key)
        candidates.append(row)

    result = []
    for path in candidates:
        names = tuple(path["names"])
        if not any(
            len(names) < len(other["names"]) and _is_prefix(names, tuple(other["names"]))
            for other in candidates
        ):
            result.append(path)
    return result


def _is_prefix(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    return long[: len(short)] == short or long[-len(short) :] == tuple(reversed(short))
