"""End-to-end scenarios against the running stack (Neo4j + Ollama + demo graph).

    uv run pytest -m live -v

Each test is one non-trivial behaviour of the system; the xfail tests document known
limitations and are expected to fail.
"""

import httpx
import pytest

from synapse_rag.config import settings
from synapse_rag.llm import get_chat, get_embedder
from synapse_rag.pipeline import Pipeline
from synapse_rag.store import GraphStore

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    store = GraphStore.from_settings()
    if not store.ping():
        pytest.skip("Neo4j is not running (docker compose up -d)")
    try:
        httpx.get(f"{settings.ollama_url}/api/version", timeout=3).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama is not running")
    if store.stats()["entities"] == 0:
        pytest.skip("Demo graph is empty (scripts/build_demo_graph.py)")
    return Pipeline(store, get_chat(), get_embedder())


def test_two_hop_compositional_question(pipeline):
    """film -> directed by -> director -> son of -> mother: needs two hops in the graph."""
    r = pipeline.ask("Who is the mother of the director of film Polish-Russian War (Film)?")
    assert r["route"] == "graph_qa"
    assert "braunek" in r["answer"].lower()
    assert r["status"] == "answered"
    assert r["evidence"], "answer must cite evidence paths/entities"
    assert any("braunek" in p["text"] for p in r["paths"])


def test_follow_up_in_russian_uses_conversation_memory(pipeline):
    """'его' must be resolved to the director from the previous turn."""
    history = [
        {"role": "user", "content": "Who is the mother of the director of film Polish-Russian War?"},
        {
            "role": "assistant",
            "content": "Małgorzata Braunek. The film was directed by Xawery Żuławski, "
            "whose mother is Małgorzata Braunek.",
        },
    ]
    r = pipeline.ask("А кто тогда его отец?", history)
    assert r["language"] == "ru"
    assert "żuławski" in r["standalone_question"].lower()
    assert "andrzej" in r["answer"].lower()


def test_router_sends_smalltalk_past_retrieval(pipeline):
    r = pipeline.ask("Привет! Расскажи анекдот про программистов")
    assert r["route"] == "smalltalk"
    assert r["graph"]["nodes"] == [] and r["paths"] == []


def test_router_answers_global_questions_from_graph_statistics(pipeline):
    r = pipeline.ask("What kinds of things does this knowledge base contain?")
    assert r["route"] == "overview"
    assert r["graph"]["nodes"], "overview shows the hub entities"


def test_validator_flags_answers_not_supported_by_the_graph(pipeline):
    """Interstellar is not in the corpus: the model may know the answer from pre-training,
    but an answer that is not grounded in the retrieved subgraph must not be 'answered'."""
    r = pipeline.ask("Who directed the film Interstellar?")
    assert r["status"] in ("not_found", "unverified")


def test_vector_baseline_has_no_paths_but_qasa_does(pipeline):
    question = "Where was the director of film Ronnie Rocket born?"
    naive = pipeline.ask(question, mode="naive", refine=False)
    qasa = pipeline.ask(question, mode="qasa", refine=False)
    assert naive["paths"] == [] and naive["seeds"] == []
    assert qasa["paths"] and qasa["seeds"]
    assert qasa["steps"] == 3 and len(qasa["graph"]["nodes"][0]["activation"]) == 4


@pytest.mark.xfail(reason="Limitation: comparisons need both branches + date arithmetic", strict=False)
def test_comparison_question_known_limitation(pipeline):
    r = pipeline.ask("Which film came out first, Blind Shaft or The Mask Of Fu Manchu?")
    assert "fu manchu" in r["answer"].lower() and r["status"] == "answered"


@pytest.mark.xfail(reason="Limitation: 3-hop kinship chain on a 4B-extracted graph", strict=False)
def test_three_hop_kinship_known_limitation(pipeline):
    r = pipeline.ask("Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?")
    assert "charles armand" in r["answer"].lower()
