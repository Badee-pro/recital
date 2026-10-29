# Recital

**Ask your contracts a question. Get an answer that cites the exact clause, or an honest "the contracts don't say".**

Recital is a retrieval-augmented generation (RAG) system for commercial contracts. It answers plain-English questions ("how fast must a data breach be reported?") using only the agreements it's given, cites the clause behind every claim, and opens that clause beside the answer so a person can check it in one click. Everything runs locally: the embedding model and the language model both run on your own machine, so contracts never leave it.

Built by [Badee Khalbouss](https://badeekhalbouss.com).

## What it does

- **Grounded answers.** The model only sees the clauses retrieved for the question and is instructed to answer from them alone.
- **Checkable citations.** Every sentence is tagged with its source clause. Tags that don't match a retrieved clause are discarded in code.
- **Refuses instead of guessing.** If the agreements don't cover the question, it says "The provided clauses don't address this."
- **Measured, not assumed.** A hand-labelled test set scores retrieval and answers, and every run is logged to MLflow.

## How it works

```
contracts (.md) ─► clause chunker ─► embeddings (bge-small, local) ─┐
                                   └► BM25 (keyword) ───────────────┤
question ─────────────────────────────────────────────────► retriever (embeddings | keyword | hybrid)
                                                                    │ top-5 clauses
                                                                    ▼
                                        local LLM (qwen2.5:7b via Ollama) ─► answer + clause citations
```

| Piece | File | Choice and why |
|---|---|---|
| Chunking | `rag/chunk.py` | Split on numbered clauses (`7.2`), not fixed-size windows; contracts already have meaningful boundaries. Each chunk carries a **contextual header** (agreement title + section) so a clause still knows which agreement it belongs to. |
| Embeddings | `rag/index.py` | `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX, CPU, no PyTorch). |
| Keyword search | `rag/retrieve.py` | BM25 with a stopword list. Good at exact terms like "AES-256" or "SOC 2". |
| Hybrid search | `rag/retrieve.py` | Reciprocal Rank Fusion of the two rankings. |
| Generation | `rag/generate.py` | `qwen2.5:7b` via Ollama at temperature 0. Clauses are labelled `[A]..[E]`; the model must tag each sentence, and the code range-checks every tag. |
| Web app | `app.py`, `web/index.html` | FastAPI backend and a single-page workspace: agreements on the left, the question thread in the middle, the source agreement on the right with the cited clause highlighted. |
| Evaluation | `eval/` | 24 hand-labelled questions (22 answerable, 2 unanswerable), retrieval and answer metrics, every run logged to **MLflow** with a trace per question. |

The corpus is 4 fictional agreements (NDA, MSA, DPA, SaaS) with 67 clauses. They deliberately overlap: three of them define a confidentiality term, a liability cap and a governing law, each with *different* values. Pulling the wrong agreement's clause is the most common way legal RAG fails, so the test set is built around it.

## Results

### Retrieval (22 answerable questions)

```
method      top-1  top-3  top-5  MRR
keyword     0.43   0.64   0.68   0.56
embeddings  0.70   0.86   0.95   0.80
hybrid      0.61   0.77   0.86   0.72
```

**Finding: hybrid search was worse than embeddings alone.** The questions are paraphrased ("if Quillon gets hacked…") while contracts use defined terms ("Personal Data Breach"), so keyword matching has little to match and its noise pushes correct results down in the fusion. Embeddings are the default; hybrid stays in the evaluation for comparison.

### Answers (embedding search, top 5)

| run | facts correct | citation correct | refused correctly |
|---|---|---|---|
| clauses labelled `[1]..[5]` | 0.95 | 0.77 | 1.00 |
| clauses labelled `[A]..[E]` | 0.91 | **0.95** | 1.00 |

**Finding: label collisions caused wrong citations.** With numeric labels, the model got the facts right but cited a neighbouring clause 4 times (e.g. 2.3 for a fact in 2.1): clause text already contains numbers like `2.1`, and the model confused them with the `[2]` label. Letter labels fixed all four. One regression remains: on q22 the model says the MSA auto-renews with "no non-renewal notice" (it's 90 days), a factual slip from a small model that a faithfulness check would catch. Raw answers are in `eval/results/`.

## Run it

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b          # ~4.7 GB, one time

python -m rag.build             # chunk + embed the contracts
python app.py                   # web app at http://localhost:8000
```

On Windows you can double-click `recital.bat` instead of `python app.py`.

Other entry points:

```bash
python ask.py "How fast must Cirrus be told about a breach?"   # command line
python -m eval.run_eval                                         # retrieval evaluation, seconds
python -m eval.run_eval --generate                              # + 24 answers, ~10 min on CPU
mlflow ui --backend-store-uri sqlite:///mlflow.db               # browse runs at http://localhost:5000
```

**Windows path limit:** if `pip` fails with `WinError 206 ... filename too long`, move the project to a shorter path (e.g. `C:\code\recital`).

## Evaluation metrics

- `recall_at_k` / `mrr`: does retrieval put the correct clause in the top k, and how high?
- `answer_match`: every expected key fact (e.g. "48", "AES-256") appears in the answer.
- `citation_hit`: at least one citation points at a correct clause.
- `citation_precision`: share of cited clauses that are correct. This is the "did it cite the right agreement?" check.
- `abstain_correct`: on unanswerable questions, does it decline instead of inventing an answer?

String matching is a deliberately simple grader; see the roadmap.

## Roadmap

1. **Reranker**: a cross-encoder over the top 20 results.
2. **Real contracts**: PDF ingestion with page-level citations, evaluated on CUAD (510 real contracts with expert labels).
3. **Party filtering**: when a question names a party ("Northwind"), search only that party's agreements.
4. **Faithfulness judge**: check each sentence against its cited clause and flag unsupported claims.

The contracts in `data/contracts/` are fictional and for demonstration only. Nothing here is legal advice.
