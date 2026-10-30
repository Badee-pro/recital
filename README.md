# Recital

**Ask your contracts a question. Get an answer that cites the exact clause, or an honest "the contracts don't say".**

Recital is a retrieval-augmented generation (RAG) system for commercial contracts. It answers plain-English questions ("how fast must a data breach be reported?") using only the agreements it's given, cites the clause behind every claim, and opens that clause beside the answer so a person can check it in one click. Everything runs locally: the embedding model and the language model both run on your own machine, so contracts never leave it.

Built by [Badee Khalbouss](https://badeekhalbouss.com).

## What it does

- **Grounded answers.** The model only sees the clauses retrieved for the question and is instructed to answer from them alone.
- **Checkable citations.** Every sentence is tagged with its source clause. Tags that don't match a retrieved clause are discarded in code.
- **Refuses instead of guessing.** If the agreements don't cover the question, it says "The provided clauses don't address this."
- **Grounding checked in code, not just in the prompt.** Off-topic questions are declined before the model is ever called, and any answer sentence without a citation is flagged.
- **Measured, not assumed.** A hand-labelled test set scores retrieval and answers, and every run is logged to MLflow.
- **AI risk review, lawyer decides.** Each agreement is checked for risky terms; flagged clauses go to a review queue where a lawyer confirms or dismisses them.
- **Upload agreements.** Add a Markdown agreement from the sidebar; it's indexed and reviewed automatically.

## The language model is one part, not the system

A 7B local model is fast and private, but it guesses, mislabels and over-flags. Recital is built so that the model never has the last word: everything it produces passes through checks in ordinary code, and every check was added because a measurement showed a specific failure.

| Decision | Made by | Why not just the model |
|---|---|---|
| Which clauses are relevant | Embedding search (`rag/retrieve.py`) | Measured against 3 search methods; the model never searches. |
| Whether a question is answerable at all | Similarity gate, 0.60 (`rag/grounding.py`) | Threshold read off measured data; off-topic questions never reach the model. |
| Whether a citation is real | Code range-checks every `[A]` label (`rag/generate.py`) | Labels pointing at clauses the model wasn't shown are dropped. |
| Whether an answer is fully sourced | Sentence-level coverage check | Caught a format slip that the accuracy score hid (see Results). |
| Which clauses a risk check may look at | Similarity gate, 0.62 (`rag/review.py`) | Stops the model judging clauses about other topics. |
| Whether a proposed risk flag survives | Second, single-clause verification call | The first pass over-flags protective clauses (see Results). |
| Whether a flag is a real risk | **A lawyer**, in the review queue | The model only proposes; dismissed flags stay dismissed. |
| Whether any of it works | Hand-labelled test sets, logged to MLflow | 24 Q&A questions and a labelled risk set, every setting compared. |

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
| Grounding checks | `rag/grounding.py` | A relevance gate (declines when no clause is close to the question) and a citation-coverage check (flags sentences with no clause label). |
| Risk review | `rag/review.py` | 7 checks, each with the market-standard position spelled out. Candidate clauses come from embedding search within one agreement, gated at 0.62; the model returns JSON per clause, labels are range-checked, and each proposed flag is re-verified on its own. |
| Review queue | `app.py`, `data/flags.json` | Reviews run on one background worker thread (the local model serves one request at a time). Lawyers' decisions survive re-reviews. Writes are atomic (temp file + rename). |
| Upload | `app.py` | Validates the file actually yields numbered clauses, rebuilds the index, swaps the retriever under a lock, and queues a review. |
| Web app | `app.py`, `frontend/` | FastAPI backend and a **React 19 + TypeScript** (strict) single-page app built with Vite: agreements on the left, the question thread or review queue in the middle, the source agreement on the right with the cited or flagged clause highlighted. No UI library; state lives in plain hooks (`useReview` polls background jobs only while one is running). |
| Evaluation | `eval/` | 24 hand-labelled questions (22 answerable, 2 unanswerable) and a hand-labelled risk set; every run logged to **MLflow**. |

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

| run | facts correct | citation correct | fully cited | refused correctly |
|---|---|---|---|---|
| clauses labelled `[1]..[5]` | 0.95 | 0.77 | – | 1.00 |
| clauses labelled `[A]..[E]` | 0.91 | **0.95** | – | 1.00 |
| + grounding checks, final prompt | **0.95** | 0.91 | 0.91 | 1.00 |

**Finding: label collisions caused wrong citations.** With numeric labels, the model got the facts right but cited a neighbouring clause 4 times (e.g. 2.3 for a fact in 2.1): clause text already contains numbers like `2.1`, and the model confused them with the `[2]` label. Letter labels fixed all four.

**Finding: the coverage check catches what the score hides.** A small prompt rewording in the final run fixed a factual error (q22 now gets both renewal terms right: 90 and 60 days) but on that answer the model wrote its sources as `(A)`/`(B)` instead of `[A]`/`[B]`. The citation-coverage check flagged exactly that answer and one sentence in q10, so the UI underlines them instead of presenting them as sourced. The parser deliberately stays strict: legal text uses `(a)`/`(b)` for sub-clauses, so accepting parentheses would risk false citations. Raw answers are in `eval/results/`.

### Grounding checks

The relevance threshold was set from data, not guessed. Cosine similarity between a question and its closest clause:

| questions | closest-clause similarity |
|---|---|
| the 22 real contract questions | 0.64 – 0.90 |
| off-topic ("weather in Toronto", "tell me a joke", "explain quantum computing", …) | 0.40 – 0.54 |

The gate is set at **0.60**, in the gap. Off-topic questions are declined without calling the model, and every real test question passes. Questions that are on-topic but unanswerable (e.g. a non-compete period the agreements never mention) still reach the model, which declines them itself: 100% on the test set.

## Run it

Requires Python 3.10+, [Node.js](https://nodejs.org) 18+ and [Ollama](https://ollama.com).

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b          # ~4.7 GB, one time

python -m rag.build             # chunk + embed the contracts
cd frontend && npm install && npm run build && cd ..   # build the React UI, one time
python app.py                   # web app at http://localhost:8000
```

On Windows you can double-click `recital.bat` instead; it builds the UI on first run.

Working on the UI: run `python app.py`, then `npm run dev` in `frontend/` for hot reload at http://localhost:5173 (API calls are proxied to :8000).

Other entry points:

```bash
python ask.py "How fast must Cirrus be told about a breach?"   # command line
python -m eval.run_eval                                         # retrieval evaluation, seconds
python -m eval.run_eval --generate                              # + 24 answers, ~10 min on CPU
python -m rag.review [doc_id]                                   # AI risk review from the command line
python -m eval.run_review_eval --ablation                       # score the risk review, ~30 min on CPU
mlflow ui --backend-store-uri sqlite:///mlflow.db               # browse runs at http://localhost:5000
```

**In the web app:** *Review queue* (top bar) lists open flags by severity. Click a flag's clause reference to open the agreement with that clause highlighted, add a note, then *Confirm risk* or *Dismiss*. *Run AI review* in the agreement viewer re-reviews one agreement. *Upload agreement* (sidebar) takes a `.md` or `.txt` file with a `# Title`, `## 1. Section` headings and numbered clauses (`1.1 ...`), like the files in `data/contracts/`; it's indexed and reviewed automatically. Uploaded agreements join the Q&A corpus, so Q&A scores measured on the original four may shift.

**Windows path limit:** if `pip` fails with `WinError 206 ... filename too long`, move the project to a shorter path (e.g. `C:\code\recital`).

## Evaluation metrics

- `recall_at_k` / `mrr`: does retrieval put the correct clause in the top k, and how high?
- `answer_match`: every expected key fact (e.g. "48", "AES-256") appears in the answer.
- `citation_hit`: at least one citation points at a correct clause.
- `citation_precision`: share of cited clauses that are correct. This is the "did it cite the right agreement?" check.
- `fully_cited`: every sentence of the answer carries a clause citation.
- `abstain_correct`: on unanswerable questions, does it decline instead of inventing an answer?

String matching is a deliberately simple grader; see the roadmap.

Risk review (`eval/run_review_eval.py`, labels in `eval/review_labels.json`):

- `recall`: share of labelled risky clauses that got a flag.
- `precision`: share of flags that land on risky clauses. Clauses labelled as judgement calls are not scored either way.
- `false_alarms`: flags on clauses that aren't risky.
- `traps_hit`: flags on clauses labelled "looks risky but is standard" (a capped price increase, a ban on training with customer data).

The labels were written by hand, not by a lawyer, and the set is small (5 agreements, 8 risky clauses, 23 labelled traps). It's for comparing settings against each other, not for claiming an absolute accuracy.

## Roadmap

1. **Reranker**: a cross-encoder over the top 20 results.
2. **Real contracts**: PDF ingestion with page-level citations, evaluated on CUAD (510 real contracts with expert labels).
3. **Party filtering**: when a question names a party ("Northwind"), search only that party's agreements.
4. **Faithfulness judge**: check each sentence against its cited clause and flag unsupported claims.
5. **Risk review from lawyers' decisions**: every confirm or dismiss in the queue is a free label. Feed them back as few-shot examples per check, and grow the risk test set from them.

The contracts in `data/contracts/` are fictional and for demonstration only. Nothing here is legal advice.
