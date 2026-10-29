"""Evaluate retrieval (and optionally generation) and log every run to MLflow.

    python -m eval.run_eval                 # retrieval only, a few seconds
    python -m eval.run_eval --generate      # + answers from the local model (Ollama)
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Retrieval metrics (answerable questions only):
  recall@k  fraction of gold clauses that appear in the top k
  mrr       1 / rank of the first gold clause (0 if absent from the top 10)

Generation metrics:
  answer_match        every expected key fact appears in the answer (string match)
  citation_hit        at least one citation points at a gold clause
  citation_precision  share of cited clauses that are gold
  fully_cited         every sentence of the answer carries a clause citation
  abstain_correct     on unanswerable questions, the model says it can't answer
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from rag.index import EMBED_MODEL
from rag.retrieve import METHODS, RRF_K, Retriever

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
ABSTAIN_PHRASE = "don't address"
CHUNKING = "clause+header"
KS = (1, 3, 5)
RANK_DEPTH = 10  # retrieval is scored on the top 10 results


def load_questions() -> list[dict]:
    lines = (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def retrieval_metrics(retriever: Retriever, questions: list[dict], method: str) -> dict:
    answerable = [q for q in questions if q["gold"]]
    totals = {f"recall_at_{k}": 0.0 for k in KS} | {"mrr": 0.0}
    misses = []
    for q in answerable:
        ranked = [c.id for c in retriever.search(q["question"], k=RANK_DEPTH, method=method)]
        gold = set(q["gold"])
        for k in KS:
            totals[f"recall_at_{k}"] += len(gold & set(ranked[:k])) / len(gold)
        first = next((i for i, cid in enumerate(ranked) if cid in gold), None)
        totals["mrr"] += 0.0 if first is None else 1.0 / (first + 1)
        if not gold & set(ranked[:5]):
            misses.append(q["id"])
    metrics = {name: v / len(answerable) for name, v in totals.items()}
    return metrics, misses


def score_answer(q: dict, answer_text: str, cited: list[str], uncited: list[str] | None = None) -> dict:
    """Generation metrics for one question (see the module docstring)."""
    text = answer_text.lower()
    if not q["gold"]:
        return {"abstain_correct": float(ABSTAIN_PHRASE in text.replace("’", "'"))}
    gold = set(q["gold"])
    return {
        "answer_match": float(all(e.lower() in text for e in q["expect"])),
        "citation_hit": float(bool(gold & set(cited))),
        "citation_precision": sum(c in gold for c in cited) / len(cited) if cited else 0.0,
        **({} if uncited is None else {"fully_cited": float(not uncited)}),
    }


def run_params(stage: str, method: str, top_k: int, llm: str) -> dict:
    # Every run logs the same keys so MLflow's compare view lines them up.
    return {"stage": stage, "method": method, "top_k": top_k, "llm": llm,
            "embed_model": EMBED_MODEL, "chunking": CHUNKING, "rrf_k": RRF_K}


def run_retrieval(retriever, questions):
    print(f"\n{'method':8s} " + " ".join(f"R@{k:<5d}" for k in KS) + " MRR    misses@5")
    for method in METHODS:
        with mlflow.start_run(run_name=f"retrieval-{method}"):
            mlflow.log_params(run_params("retrieval", method, RANK_DEPTH, "none"))
            metrics, misses = retrieval_metrics(retriever, questions, method)
            mlflow.log_metrics(metrics)
        print(f"{method:8s} " + " ".join(f"{metrics[f'recall_at_{k}']:<7.2f}" for k in KS)
              + f"{metrics['mrr']:<7.2f}{', '.join(misses) or '-'}")


def answer_traced(retriever, question: dict, method: str, k: int):
    """Retrieve + answer one question, recorded as an MLflow trace (GenAI > Traces)."""
    from rag.generate import MODEL, answer

    with mlflow.start_span(name=f"rag-{question['id']}", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"question": question["question"]})
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": question["question"], "method": method, "k": k})
            chunks = retriever.search(question["question"], k=k, method=method)
            span.set_outputs([{"id": c.id, "section": c.section, "text": c.text} for c in chunks])
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as span:
            span.set_inputs({"model": MODEL, "question": question["question"],
                             "clauses": [c.id for c in chunks]})
            ans = answer(question["question"], chunks, top_similarity=retriever.top_similarity(question["question"]))
            span.set_outputs({"answer": ans.text, "cited": ans.cited_chunk_ids, "usage": ans.usage,
                              "gated": ans.gated, "uncited": ans.uncited})
        root.set_outputs({"answer": ans.text, "cited": ans.cited_chunk_ids})
    return ans


def run_generation(retriever, questions, method: str, k: int):
    from rag.generate import MODEL

    # The run is opened before generating so its duration covers the slow part
    # and each question's trace is linked to it.
    with mlflow.start_run(run_name=f"generation-local-{method}-k{k}"):
        mlflow.log_params(run_params("generation", method, k, MODEL))

        rows, scores = [], {"answer_match": [], "citation_hit": [], "citation_precision": [],
                            "fully_cited": [], "abstain_correct": []}
        for q in questions:
            ans = answer_traced(retriever, q, method, k)
            cited = ans.cited_chunk_ids
            for name, value in score_answer(q, ans.text, cited, ans.uncited).items():
                scores[name].append(value)
            rows.append({"id": q["id"], "question": q["question"], "answer": ans.text,
                         "cited": cited, "gold": q["gold"], "usage": ans.usage,
                         "top_similarity": ans.top_similarity, "gated": ans.gated, "uncited": ans.uncited})
            print(f"  {q['id']}  cited={cited}")

        metrics = {name: sum(v) / len(v) for name, v in scores.items() if v}
        print("\n" + "\n".join(f"{name:20s} {v:.2f}" for name, v in metrics.items()))

        # Save answers locally first so a tracking failure can never lose a run.
        RESULTS_DIR.mkdir(exist_ok=True)
        # The default setup writes answers_local.json, which the web app's scores read;
        # other settings get their own file so an experiment can't overwrite the baseline.
        suffix = "" if (method, k) == ("dense", 5) else f"_{method}_k{k}"
        answers_path = RESULTS_DIR / f"answers_local{suffix}.json"
        answers_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"answers saved to {answers_path.relative_to(EVAL_DIR.parent)}")

        mlflow.log_metrics(metrics)
        try:
            mlflow.log_artifact(str(answers_path))
        except OSError as e:
            # Windows' 260-char path limit bites when the project sits in a deep folder.
            print(f"warning: couldn't attach answers to the MLflow run ({e.__class__.__name__}); "
                  f"metrics were logged and answers are in {answers_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--method", choices=METHODS, default="dense")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--skip-retrieval", action="store_true", help="don't re-log the retrieval comparison runs")
    args = p.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("contract-rag")

    retriever = Retriever()
    questions = load_questions()
    if not args.skip_retrieval:
        run_retrieval(retriever, questions)
    if args.generate:
        print(f"\nGenerating answers ({args.method}, k={args.k})...")
        run_generation(retriever, questions, args.method, args.k)


if __name__ == "__main__":
    main()
