"""Recital -- a small web app over the contract RAG pipeline.

    python app.py            # then open http://localhost:8000

The page searches first (fast) and shows the matching clauses, then asks the
model for a cited answer (slow on a local model), so there's something to look
at while the answer is written.
"""

import json
import queue
import re
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eval.run_eval import RESULTS_DIR, load_questions, retrieval_metrics, score_answer
from rag import review
from rag.chunk import chunk_document
from rag.generate import MODEL, answer
from rag.grounding import MIN_SIMILARITY
from rag.index import CORPUS_DIR, EMBED_MODEL, build_index, load_index
from rag.retrieve import METHODS, Retriever

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "frontend" / "dist"  # the React UI; build it with `npm run build` in frontend/

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


# AI risk reviews run one at a time on a background thread (the local model can
# only answer one request at a time anyway). Jobs: doc_id -> {state, done, total}.
review_jobs: dict[str, dict] = {}
review_queue: queue.Queue[str] = queue.Queue()
index_lock = threading.Lock()


def review_worker():
    while True:
        doc_id = review_queue.get()
        job = review_jobs[doc_id]
        job["state"] = "running"
        try:
            with index_lock:
                chunks, embeddings = retriever.chunks, retriever.embeddings
            review.review_document(chunks, embeddings, doc_id,
                                   on_progress=lambda i, n: job.update(done=i, total=n))
            job.update(state="done", done=job["total"])
        except Exception as e:  # usually Ollama not running
            job.update(state="error", error=str(e))


def queue_review(doc_id: str):
    if review_jobs.get(doc_id, {}).get("state") in ("queued", "running"):
        return
    review_jobs[doc_id] = {"state": "queued", "done": 0, "total": len(review.CHECKS)}
    review_queue.put(doc_id)


threading.Thread(target=review_worker, daemon=True).start()


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    method: str = "dense"
    k: int = Field(default=5, ge=1, le=8)


class Upload(BaseModel):
    filename: str = Field(min_length=1, max_length=120)
    markdown: str = Field(min_length=20, max_length=500_000)


class Decision(BaseModel):
    status: str
    reviewer: str = Field(default="", max_length=80)
    note: str = Field(default="", max_length=2000)


def search(q: Query):
    if q.method not in METHODS:
        raise HTTPException(400, f"method must be one of {METHODS}")
    return retriever.search(q.question, k=q.k, method=q.method)


@app.get("/")
def index():
    if not (WEB_DIR / "index.html").exists():
        return HTMLResponse("<p style='font-family:monospace'>The UI isn't built yet. Run "
                            "<code>cd frontend && npm install && npm run build</code>, then reload.</p>", 503)
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/meta")
def meta():
    counts = {}
    for c in retriever.chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    open_flags = {}
    for f in review.load_flags():
        if f["status"] == "open":
            open_flags[f["doc_id"]] = open_flags.get(f["doc_id"], 0) + 1
    return {
        "methods": METHODS,
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "contracts": [{"id": d, "title": doc_titles[d], "clauses": n, "open_flags": open_flags.get(d, 0)}
                      for d, n in counts.items()],
        "examples": EXAMPLES,
    }


@app.post("/api/contracts")
def upload_contract(u: Upload):
    """Add an agreement (Markdown: '# Title', '## 1. Section', numbered clauses like '1.1 ...'),
    re-index the corpus, and queue an AI risk review."""
    global retriever, doc_titles
    doc_id = re.sub(r"[^a-z0-9]+", "_", Path(u.filename).stem.lower()).strip("_")[:60]
    if not doc_id:
        raise HTTPException(400, "Give the file a name with letters or numbers.")
    path = CORPUS_DIR / f"{doc_id}.md"
    if path.exists():
        raise HTTPException(409, f"An agreement named '{doc_id}' already exists.")

    path.write_text(u.markdown.replace("\r\n", "\n"), encoding="utf-8")
    clauses = chunk_document(path)
    if not clauses:
        path.unlink()
        raise HTTPException(400, "No numbered clauses found. Use '## 1. Section' headings and clauses like '1.1 ...'.")

    with index_lock:
        build_index()
        chunks, embeddings = load_index()
        retriever = Retriever(chunks, embeddings)
        doc_titles = {c.doc_id: c.doc_title for c in retriever.chunks}
    queue_review(doc_id)
    return {"id": doc_id, "title": clauses[0].doc_title, "clauses": len(clauses)}


@app.get("/api/flags")
def flags():
    return {"flags": review.load_flags(), "checks": review.CHECK_TITLES}


@app.post("/api/flags/{flag_id}")
def decide_flag(flag_id: str, d: Decision):
    if d.status not in ("open", "confirmed", "dismissed"):
        raise HTTPException(400, "status must be open, confirmed or dismissed")
    try:
        return review.decide(flag_id, d.status, d.reviewer, d.note)
    except KeyError:
        raise HTTPException(404, "unknown flag")


@app.post("/api/review/{doc_id}")
def start_review(doc_id: str):
    if doc_id not in doc_titles:
        raise HTTPException(404, "unknown contract")
    queue_review(doc_id)
    return review_jobs[doc_id]


@app.get("/api/review/status")
def review_status():
    return review_jobs


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
        ans = answer(q.question, chunks, top_similarity=retriever.top_similarity(q.question))
    except Exception:  # Ollama not running, or the model isn't pulled
        raise HTTPException(503, "The model couldn't answer. Is Ollama running? Open the Ollama app and try again.")
    return {
        "ms": round((time.perf_counter() - start) * 1000),
        "model": MODEL,
        "answer": ans.marked_text or ans.text,
        "cited": ans.cited_chunk_ids,
        "usage": ans.usage,
        "grounding": {"top_similarity": round(ans.top_similarity, 3), "threshold": MIN_SIMILARITY,
                      "gated": ans.gated, "uncited": ans.uncited},
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


# Vite puts the bundled JS and CSS under dist/assets/. Mounted last so it can't shadow /api routes.
app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets", check_dir=False), name="assets")


if __name__ == "__main__":
    print("Recital running at http://localhost:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
