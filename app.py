"""Recital -- a small web app over the contract RAG pipeline.

    python app.py            # then open http://localhost:8000

The page searches first (fast) and shows the matching clauses, then asks the
model for a cited answer (slow on a local model), so there's something to look
at while the answer is written.
"""

import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from eval.run_eval import RESULTS_DIR, load_questions, retrieval_metrics, score_answer
from rag.generate import MODEL, answer
from rag.index import CORPUS_DIR, EMBED_MODEL
from rag.retrieve import METHODS, Retriever

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

EXAMPLES = [
    "If Quillon gets hacked, how fast must Cirrus be told?",
    "Which agreements renew automatically, and what notice is needed?",
    "Can Orbital use Vantage's data to train models for other clients?",
    "What happens if the NDA recipient gets a subpoena?",
    "What is the CEO's salary at Orbital Insights?",
]

app = FastAPI(title="Recital")
retriever = Retriever()
doc_titles = {c.doc_id: c.doc_title for c in retriever.chunks}


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    method: str = "dense"
    k: int = Field(default=5, ge=1, le=8)


def search(q: Query):
    if q.method not in METHODS:
        raise HTTPException(400, f"method must be one of {METHODS}")
    return retriever.search(q.question, k=q.k, method=q.method)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/meta")
def meta():
    counts = {}
    for c in retriever.chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    return {
        "methods": METHODS,
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "contracts": [{"id": d, "title": doc_titles[d], "clauses": n} for d, n in counts.items()],
        "examples": EXAMPLES,
    }


@app.get("/api/contracts/{doc_id}")
def contract(doc_id: str):
    if doc_id not in doc_titles:
        raise HTTPException(404, "unknown contract")
    return {"id": doc_id, "title": doc_titles[doc_id],
            "markdown": (CORPUS_DIR / f"{doc_id}.md").read_text(encoding="utf-8")}


@app.post("/api/search")
def search_endpoint(q: Query):
    start = time.perf_counter()
    chunks = search(q)
    return {
        "ms": round((time.perf_counter() - start) * 1000),
        "clauses": [{"n": i + 1, "id": c.id, "doc_id": c.doc_id, "doc_title": c.doc_title,
                     "section": c.section, "clause": c.clause, "text": c.text}
                    for i, c in enumerate(chunks)],
    }


@app.post("/api/answer")
def answer_endpoint(q: Query):
    chunks = search(q)  # deterministic, so this matches what /api/search showed
    start = time.perf_counter()
    try:
        ans = answer(q.question, chunks)
    except Exception:  # Ollama not running, or the model isn't pulled
        raise HTTPException(503, "The model couldn't answer. Is Ollama running? Open the Ollama app and try again.")
    return {
        "ms": round((time.perf_counter() - start) * 1000),
        "model": MODEL,
        "answer": ans.marked_text or ans.text,
        "cited": ans.cited_chunk_ids,
        "usage": ans.usage,
    }


@app.get("/api/stats")
def stats():
    questions = load_questions()
    retrieval = {m: retrieval_metrics(retriever, questions, m)[0] for m in METHODS}

    generation = {}
    by_id = {q["id"]: q for q in questions}
    for path in sorted(RESULTS_DIR.glob("answers_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != len(questions):  # skip partial or older-format runs
            continue
        totals: dict[str, list[float]] = {}
        for row in rows:
            for name, v in score_answer(by_id[row["id"]], row["answer"], row["cited"]).items():
                totals.setdefault(name, []).append(v)
        generation[path.stem.removeprefix("answers_")] = {n: sum(v) / len(v) for n, v in totals.items()}

    return {"n_questions": len(questions), "retrieval": retrieval, "generation": generation}


if __name__ == "__main__":
    print("Recital running at http://localhost:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
