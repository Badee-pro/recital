"""Evaluate the AI risk review against hand labels and log each run to MLflow.

    python -m eval.run_review_eval                  # the shipped setup (verify), ~10 min on CPU
    python -m eval.run_review_eval --ablation       # + with each safeguard turned off
    python -m eval.run_review_eval --setup verify   # one setup

Corpus: the four agreements plus eval/review_fixtures/ (a deliberately one-sided
vendor agreement; the four real ones are fairly balanced, so on their own they'd
test false alarms but barely test recall). Fixtures are not part of the Q&A corpus.

Metrics (labels in eval/review_labels.json):
  recall       share of risky clauses that got a flag
  precision    share of flags that are on risky clauses ('either' clauses not scored)
  false_alarms flags on clauses that aren't risky
  traps_hit    flags on labelled 'standard' clauses that only look risky
"""

import argparse
import json
import os
from pathlib import Path

import mlflow

from rag.chunk import chunk_document, load_corpus
from rag.generate import MODEL
from rag.index import CORPUS_DIR, EMBED_MODEL, embed_passages
from rag.review import CHECKS, CLAUSES_PER_CHECK, MIN_CHECK_SIMILARITY, find_flags

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
FIXTURES_DIR = EVAL_DIR / "review_fixtures"

# name -> (similarity gate, verification call)
CONFIGS = {
    "first_pass_only": (False, False),
    "gate": (True, False),
    "verify": (False, True),
    "gate_verify": (True, True),
}


def load_labels() -> dict:
    return json.loads((EVAL_DIR / "review_labels.json").read_text(encoding="utf-8"))["documents"]


def score(labels: dict, flags: list[dict]) -> tuple[dict, list[dict]]:
    """Metrics over all documents, plus one row per flag or missed risk for the results file."""
    flagged = {f["clause_id"]: f for f in flags}
    risky = {cid: check for d in labels.values() for cid, check in d["risky"].items()}
    standard = {cid for d in labels.values() for cid in d["standard"]}
    either = {cid for d in labels.values() for cid in d["either"]}

    hits = [cid for cid in risky if cid in flagged]
    scored = [cid for cid in flagged if cid not in either]
    false = [cid for cid in scored if cid not in risky]
    metrics = {
        "recall": len(hits) / len(risky),
        "precision": len(hits) / len(scored) if scored else 0.0,
        "false_alarms": float(len(false)),
        "traps_hit": float(len(standard & set(flagged))),
        "flags": float(len(flags)),
    }
    rows = [{"clause_id": cid, "outcome": "caught" if cid in risky else "either" if cid in either
             else "false_alarm", "check": f["check"], "severity": f["severity"], "reason": f["reason"]}
            for cid, f in flagged.items()]
    rows += [{"clause_id": cid, "outcome": "missed", "check": check} for cid, check in risky.items()
             if cid not in flagged]
    return metrics, rows


def run(name: str, chunks, embeddings, labels):
    gate, verify = CONFIGS[name]
    with mlflow.start_run(run_name=f"review-{name}"):
        mlflow.log_params({"stage": "review", "llm": MODEL, "embed_model": EMBED_MODEL, "gate": gate,
                           "verify": verify, "min_check_similarity": MIN_CHECK_SIMILARITY if gate else 0,
                           "clauses_per_check": CLAUSES_PER_CHECK, "checks": len(CHECKS)})
        flags = []
        for doc_id in labels:
            print(f"  {doc_id}")
            flags += find_flags(chunks, embeddings, doc_id, gate=gate, verify=verify)
        metrics, rows = score(labels, flags)

        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"review_{name}.json"
        path.write_text(json.dumps(sorted(rows, key=lambda r: r["clause_id"]), indent=2), encoding="utf-8")
        mlflow.log_metrics(metrics)
        try:
            mlflow.log_artifact(str(path))
        except OSError as e:  # Windows' 260-char path limit
            print(f"warning: couldn't attach results to the MLflow run ({e.__class__.__name__})")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", action="store_true", help="also run with each safeguard turned off")
    p.add_argument("--setup", choices=CONFIGS, help="run one setup only")
    args = p.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("contract-rag")

    chunks = load_corpus(CORPUS_DIR)
    for path in sorted(FIXTURES_DIR.glob("*.md")):
        chunks += chunk_document(path)
    embeddings = embed_passages([c.indexed_text for c in chunks])
    labels = load_labels()

    results = {}
    for name in (CONFIGS if args.ablation else [args.setup or "verify"]):
        print(f"\n{name}")
        results[name] = run(name, chunks, embeddings, labels)

    print(f"\n{'setup':16s} recall  precision  false_alarms  traps_hit  flags")
    for name, m in results.items():
        print(f"{name:16s} {m['recall']:<7.2f} {m['precision']:<10.2f} {m['false_alarms']:<13.0f} "
              f"{m['traps_hit']:<10.0f} {m['flags']:.0f}")


if __name__ == "__main__":
    main()
