"""Small local evaluation of the app on the demo-graph questions (2WikiMultiHopQA subset).

    uv run python scripts/evaluate_local.py --questions 60 --modes qasa naive

Every question goes through the full pipeline (analyzer, retrieval, generation, validation,
refinement). Results: notebooks/results/local-eval.csv, summarized in
notebooks/evaluation-summary.ipynb.
"""

import argparse
import csv
import time

from tqdm import tqdm

from synapse_rag.config import ROOT, settings
from synapse_rag.datasets import load_examples
from synapse_rag.llm import get_chat, get_embedder
from synapse_rag.metrics import exact_match, f1_score
from synapse_rag.pipeline import Pipeline
from synapse_rag.store import GraphStore

OUTPUT = ROOT / "notebooks" / "results" / "local-eval.csv"
FIELDS = [
    "question_id", "type", "mode", "question", "gold", "answer", "status", "em", "f1",
    "grounded", "refined", "paths", "seeds", "seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--modes", nargs="+", default=["qasa", "naive"])
    args = parser.parse_args()

    examples = load_examples(settings.data_dir / "2wiki_demo.json")[: args.questions]
    pipeline = Pipeline(GraphStore.from_settings(), get_chat(), get_embedder())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for example in tqdm(examples, desc="Questions"):
            for mode in args.modes:
                started = time.perf_counter()
                try:
                    r = pipeline.ask(example["question"], mode=mode)
                except Exception as exc:  # keep the run going, record the failure
                    r = {"answer": "", "status": f"error: {exc}"[:80], "grounded": False,
                         "refined": False, "paths": [], "seeds": []}
                writer.writerow({
                    "question_id": example["id"],
                    "type": example["type"],
                    "mode": mode,
                    "question": example["question"],
                    "gold": example["answer"],
                    "answer": r["answer"],
                    "status": r["status"],
                    "em": exact_match(r["answer"], example["answer"]),
                    "f1": round(f1_score(r["answer"], example["answer"]), 4),
                    "grounded": r["grounded"],
                    "refined": r["refined"],
                    "paths": len(r["paths"]),
                    "seeds": len(r["seeds"]),
                    "seconds": round(time.perf_counter() - started, 2),
                })
                f.flush()
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
