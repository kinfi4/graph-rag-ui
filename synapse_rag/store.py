"""Neo4j access: schema, batched writes for ingestion, read helpers for the API.

Graph schema (same as the QASA research):
    (:Document {title, source})-[:HAS_PARAGRAPH]->(:Paragraph {id, text, doc_title})
    (:Paragraph|Document)-[:MENTIONS]->(:Entity)
    (:Entity {name, type, description, description_embedding, description_hash, doc_count})
        -[:RELATES_TO {relation}]->(:Entity)
"""

from typing import Any

from neo4j import Driver, GraphDatabase, ManagedTransaction

from synapse_rag.config import settings
from synapse_rag.extraction import ExtractionResult

ENTITY_INDEX = "entity_description_embeddings"


class GraphStore:
    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    @classmethod
    def from_settings(cls) -> "GraphStore":
        auth = (settings.neo4j_user, settings.neo4j_password)
        # Notifications are schema hints ("property does not exist yet") that spam the logs.
        driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=auth, notifications_min_severity="OFF"
        )
        return cls(driver)

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(query, parameters_=params)
        return [record.data() for record in records]

    def ping(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ---------- schema ----------

    def ensure_schema(self, dim: int) -> None:
        for statement in (
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT document_title IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.title IS UNIQUE",
            "CREATE CONSTRAINT paragraph_id IF NOT EXISTS "
            "FOR (p:Paragraph) REQUIRE p.id IS UNIQUE",
            f"""CREATE VECTOR INDEX {ENTITY_INDEX} IF NOT EXISTS
                FOR (e:Entity) ON (e.description_embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {int(dim)},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
        ):
            self.run(statement)

    def clear(self) -> None:
        # CALL ... IN TRANSACTIONS only works in an auto-commit transaction.
        with self.driver.session() as session:
            session.run(
                "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"
            ).consume()

    # ---------- ingestion ----------

    def write_chunk(
        self,
        title: str,
        source: str,
        paragraph_id: str,
        text: str,
        extraction: ExtractionResult,
    ) -> None:
        """Write one validated chunk (document, paragraph, entities, relations) atomically."""
        text_lower = text.lower()
        names = [e.name for e in extraction.entities]
        payload = {
            "title": title,
            "source": source,
            "pid": paragraph_id,
            "text": text,
            "entities": [e.model_dump() for e in extraction.entities],
            "triples": [t.model_dump() for t in extraction.triples],
            "in_text": [n for n in names if n in text_lower],
            "not_in_text": [n for n in names if n not in text_lower],
        }
        with self.driver.session() as session:
            session.execute_write(_write_chunk_tx, payload)

    def update_doc_counts(self) -> None:
        self.run(
            """
            MATCH (e:Entity)
            CALL (e) {
                OPTIONAL MATCH (d:Document)-[:HAS_PARAGRAPH]->(:Paragraph)-[:MENTIONS]->(e)
                WITH collect(DISTINCT d.title) AS via_paragraph
                OPTIONAL MATCH (d2:Document)-[:MENTIONS]->(e)
                WITH via_paragraph, collect(DISTINCT d2.title) AS direct
                RETURN size(via_paragraph + [t IN direct WHERE NOT t IN via_paragraph])
                    AS doc_count
            }
            SET e.doc_count = doc_count
            """
        )

    def entities_for_embedding(self) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (e:Entity) WHERE e.description IS NOT NULL
            RETURN e.name AS name, e.description AS description,
                   e.description_hash AS description_hash
            """
        )

    def set_embeddings(self, rows: list[dict[str, Any]]) -> None:
        for start in range(0, len(rows), 200):
            self.run(
                """
                UNWIND $rows AS row
                MATCH (e:Entity {name: row.name})
                SET e.description_embedding = row.embedding,
                    e.description_hash = row.description_hash
                """,
                rows=rows[start : start + 200],
            )

    # ---------- reads for the API ----------

    def stats(self) -> dict[str, int]:
        rows = self.run(
            """
            CALL () { MATCH (d:Document) RETURN count(d) AS documents }
            CALL () { MATCH (p:Paragraph) RETURN count(p) AS paragraphs }
            CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL () { MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS relations }
            CALL () { MATCH ()-[m:MENTIONS]->() RETURN count(m) AS mentions }
            RETURN documents, paragraphs, entities, relations, mentions
            """
        )
        return rows[0]

    def documents(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (d:Document)
            WITH d ORDER BY d.added_at DESC, d.title LIMIT $limit
            RETURN d.title AS title, d.source AS source,
                   COUNT { (d)-[:HAS_PARAGRAPH]->()-[:MENTIONS]->() } +
                   COUNT { (d)-[:MENTIONS]->() } AS mentions
            """,
            limit=limit,
        )

    def entity_details(self, names: list[str]) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (e:Entity) WHERE e.name IN $names
            OPTIONAL MATCH (d:Document)-[:HAS_PARAGRAPH]->(:Paragraph)-[:MENTIONS]->(e)
            WITH e, collect(DISTINCT d.title)[..3] AS documents
            RETURN e.name AS name, e.type AS type, e.description AS description,
                   coalesce(e.doc_count, 0) AS doc_count, documents,
                   COUNT { (e)-[:RELATES_TO]-() } AS degree
            """,
            names=names,
        )

    def edges_between(self, names: list[str]) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE a.name IN $names AND b.name IN $names
            RETURN a.name AS source, b.name AS target, r.relation AS relation
            """,
            names=names,
        )

    def supporting_paragraphs(self, names: list[str], limit: int = 4) -> list[dict[str, Any]]:
        """Source paragraphs that mention the most evidence entities (provenance for the UI)."""
        return self.run(
            """
            MATCH (p:Paragraph)-[:MENTIONS]->(e:Entity) WHERE e.name IN $names
            WITH p, collect(DISTINCT e.name) AS hits
            ORDER BY size(hits) DESC LIMIT $limit
            RETURN p.doc_title AS title, p.text AS text, hits
            """,
            names=names,
            limit=limit,
        )

    def hub_entities(self, limit: int = 25) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (e:Entity)
            WITH e, COUNT { (e)-[:RELATES_TO]-() } AS degree
            ORDER BY degree DESC, e.doc_count DESC LIMIT $limit
            RETURN e.name AS name, e.type AS type, e.description AS description, degree
            """,
            limit=limit,
        )

    def type_counts(self) -> list[dict[str, Any]]:
        return self.run(
            "MATCH (e:Entity) RETURN coalesce(e.type, 'UNKNOWN') AS type, count(*) AS n "
            "ORDER BY n DESC"
        )


def _write_chunk_tx(tx: ManagedTransaction, p: dict[str, Any]) -> None:
    tx.run(
        """
        MERGE (d:Document {title: $title})
        ON CREATE SET d.source = $source, d.added_at = datetime()
        MERGE (para:Paragraph {id: $pid})
        ON CREATE SET para.text = $text, para.doc_title = $title
        MERGE (d)-[:HAS_PARAGRAPH]->(para)
        """,
        title=p["title"], source=p["source"], pid=p["pid"], text=p["text"],
    )
    tx.run(
        """
        UNWIND $entities AS ent
        MERGE (e:Entity {name: ent.name})
        ON CREATE SET e.type = ent.type, e.description = ent.description
        ON MATCH SET
            e.type = CASE
                WHEN e.type IS NULL OR (e.type = 'CONCEPT' AND ent.type <> 'CONCEPT')
                THEN ent.type ELSE e.type END,
            e.description = CASE
                WHEN e.description IS NULL OR e.description = '' THEN ent.description
                WHEN ent.description = '' OR e.description CONTAINS ent.description
                THEN e.description
                ELSE e.description + ' | ' + ent.description END
        """,
        entities=p["entities"],
    )
    tx.run(
        """
        UNWIND $triples AS t
        MATCH (s:Entity {name: t.subject})
        MATCH (o:Entity {name: t.object})
        MERGE (s)-[r:RELATES_TO]->(o)
        ON CREATE SET r.relation = t.relation
        ON MATCH SET r.relation = CASE
            WHEN r.relation CONTAINS t.relation THEN r.relation
            ELSE r.relation + ' | ' + t.relation END
        """,
        triples=p["triples"],
    )
    tx.run(
        """
        MATCH (para:Paragraph {id: $pid})
        UNWIND $names AS name
        MATCH (e:Entity {name: name})
        MERGE (para)-[:MENTIONS]->(e)
        """,
        pid=p["pid"], names=p["in_text"],
    )
    tx.run(
        """
        MATCH (d:Document {title: $title})
        UNWIND $names AS name
        MATCH (e:Entity {name: name})
        MERGE (d)-[:MENTIONS]->(e)
        """,
        title=p["title"], names=p["not_in_text"],
    )
