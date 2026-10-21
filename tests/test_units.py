"""Fast tests for the deterministic parts of the pipeline (no Neo4j / LLM needed)."""

from synapse_rag.extraction import (
    EXTRACTION_SCHEMA,
    Entity,
    ExtractionResult,
    Triple,
    normalize_name,
    validate_extraction,
)
from synapse_rag.ingest import canonical_description, chunk_text
from synapse_rag.metrics import exact_match, f1_score
from synapse_rag.pipeline import Context, build_context, is_grounded, validate_evidence
from synapse_rag.retrieval import RetrievalParams, RetrievalResult, deduplicate_paths


def test_validation_drops_junk_and_repairs_dangling_triples():
    raw = ExtractionResult(
        entities=[
            Entity(name="Xawery  Żuławski", type="PERSON", description="Polish film director"),
            Entity(name="xawery żuławski", type="PERSON", description="Born in Warsaw"),
            Entity(name="He", type="PERSON", description="pronoun"),
            Entity(name="1971", type="EVENT", description="a year"),
        ],
        triples=[
            Triple(subject="Xawery Żuławski", relation="Son Of", object="Małgorzata Braunek"),
            Triple(subject="He", relation="born in", object="Warsaw"),
            Triple(subject="Xawery Żuławski", relation="is", object="xawery żuławski"),
            Triple(subject="Xawery Żuławski", relation="son of", object="Małgorzata Braunek"),
        ],
    )
    clean, report = validate_extraction(raw, "Xawery Żuławski")

    names = {e.name for e in clean.entities}
    assert names == {"xawery żuławski", "małgorzata braunek"}
    merged = next(e for e in clean.entities if e.name == "xawery żuławski")
    assert "Polish film director" in merged.description and "Warsaw" in merged.description
    assert [(t.subject, t.relation, t.object) for t in clean.triples] == [
        ("xawery żuławski", "son of", "małgorzata braunek")
    ]
    assert report.repaired_endpoints == ["małgorzata braunek"]
    assert set(report.dropped_entities) == {"He", "1971"}
    assert report.dropped_triples == 3  # pronoun subject, self-loop, duplicate


def test_normalize_name():
    assert normalize_name('  "The  Beatles". ') == "the beatles"


def test_bounded_extraction_schema_limits_lists_and_strings():
    assert EXTRACTION_SCHEMA["properties"]["entities"]["maxItems"] == 20
    assert EXTRACTION_SCHEMA["$defs"]["Entity"]["properties"]["description"]["maxLength"] == 180
    # the storage model itself stays unbounded, so old cache entries still load
    assert "maxItems" not in ExtractionResult.model_json_schema()["properties"]["entities"]


def test_chunk_text_packs_paragraphs_and_splits_long_ones():
    text = "First paragraph.\n\nSecond paragraph.\n\n" + "A long sentence here. " * 30
    chunks = chunk_text(text, max_chars=200)
    assert chunks[0] == "First paragraph.\n\nSecond paragraph."
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks).count("A long sentence here.") == 30


def test_canonical_description_is_order_independent():
    a = canonical_description("bern", "Swiss city | Capital of Switzerland")
    b = canonical_description("bern", "Capital of Switzerland | Swiss city | Swiss city")
    assert a == b == "**Bern**: Capital of Switzerland | Swiss city"


def test_deduplicate_paths_removes_reversed_and_prefix_paths():
    rows = [
        {"names": ["a", "b", "c"], "edges": [], "path_weight": 0.9},
        {"names": ["c", "b", "a"], "edges": [], "path_weight": 0.8},  # reversed duplicate
        {"names": ["a", "b"], "edges": [], "path_weight": 0.7},  # prefix of a-b-c
        {"names": ["a", "d"], "edges": [], "path_weight": 0.6},
    ]
    assert [r["names"] for r in deduplicate_paths(rows)] == [["a", "b", "c"], ["a", "d"]]


def test_build_context_numbers_paths_and_renders_direction():
    retrieval = RetrievalResult(
        mode="qasa",
        params=RetrievalParams(),
        paths=[
            {
                "names": ["polish-russian war", "xawery żuławski", "małgorzata braunek"],
                "edges": [
                    {"relation": "directed by", "forward": True},
                    {"relation": "mother of", "forward": False},
                ],
                "path_weight": 0.5,
            }
        ],
        entities=[
            {"name": "xawery żuławski", "type": "PERSON", "description": "Director"},
            {"name": "no description", "type": "PERSON", "description": None},
        ],
    )
    context = build_context(retrieval)
    assert context.paths_text == (
        "[P1] polish-russian war --[directed by]--> xawery żuławski "
        "<--[mother of]-- małgorzata braunek"
    )
    assert context.entities_text == "[E1] xawery żuławski (PERSON): Director"


def test_validate_evidence_keeps_only_known_ids():
    context = Context(path_ids={"P1": {}, "P2": {}}, entity_ids={"E1": {}})
    raw = ["P1", "[P2]", "2", "e1", "P9", "X1", "P1"]
    assert validate_evidence(raw, context) == ["P1", "P2", "E1"]


def test_grounding_check():
    context = "xawery żuławski --[son of]--> małgorzata braunek"
    assert is_grounded("Małgorzata Braunek", context)
    assert is_grounded("yes", context)
    assert not is_grounded("Christopher Nolan", context)
    assert not is_grounded("", context)


def test_retrieval_params_expand_for_refinement():
    wider = RetrievalParams().expanded()
    assert wider.max_steps == 4 and wider.threshold == 0.005 and wider.top_k_paths == 45


def test_answer_metrics():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert f1_score("Missoula, Montana", "Missoula") == 2 / 3
    assert f1_score("Paris", "London") == 0.0
