"""AI risk review: check each agreement against a checklist and flag clauses
for a lawyer to confirm or dismiss.

For every (agreement, check) pair, the agreement's own clauses are ranked by
embedding similarity to the check, the top few are labelled [A]..[C], and the
model is asked whether any of them raises the risk. As with answers, the label
is range-checked in code: a flag must point at a clause the model was shown,
or it is dropped. The model only proposes flags; a lawyer decides.

    python -m rag.review                      # review every agreement
    python -m rag.review msa_northwind_quillon
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunk import Chunk
from .generate import MODEL
from .index import ROOT, embed_query

FLAGS_PATH = ROOT / "data" / "flags.json"
CLAUSES_PER_CHECK = 3
# Clauses less similar than this to a check are about something else and never
# reach the model. Measured on this corpus: on-topic clauses score 0.63-0.76;
# off-topic ones that slipped into the top 3 (an invoice-dispute clause for
# "breach notice") score 0.56-0.61.
MIN_CHECK_SIMILARITY = 0.62
SEVERITIES = ("high", "medium", "low")

# `query` finds candidate clauses; `ask` is what the model judges them against;
# `standard` is the market-normal position, which is not worth a lawyer's time.
CHECKS = [
    {"id": "liability", "title": "Uncapped or one-sided liability",
     "query": "limitation of liability cap on damages, exclusions from the cap",
     "ask": "Is a party's liability uncapped, capped below 12 months' fees, or does the cap apply to one party only?",
     "standard": "A mutual cap of 12 months' fees or more, a higher cap for data breaches, and a mutual exclusion of "
                 "indirect or consequential damages are all standard."},
    {"id": "auto_renewal", "title": "Automatic renewal",
     "query": "term renews automatically unless notice of non-renewal is given",
     "ask": "Does the agreement renew automatically unless a party gives notice of non-renewal?",
     "standard": "Fixed terms that end without renewing, or renewals that need both parties to agree."},
    {"id": "termination", "title": "One-sided termination",
     "query": "termination for convenience, right to terminate without cause, suspension",
     "ask": "Can one party terminate or suspend at will (without cause) while the other cannot?",
     "standard": "Either party terminating for material breach after a cure period, or both parties having the same right."},
    {"id": "indemnity", "title": "Broad or one-sided indemnity",
     "query": "indemnify, defend and hold harmless against third-party claims",
     "ask": "Must a party indemnify against all claims or losses (not just third-party IP claims), without limit?",
     "standard": "A provider defending the customer against third-party IP infringement claims protects the customer."},
    {"id": "data_use", "title": "Provider use of customer data",
     "query": "provider may use customer data, aggregated data, analytics, training machine learning models",
     "ask": "Does a clause PERMIT the provider to use customer data for its own purposes (product improvement, "
            "analytics, benchmarking, training models)?",
     "standard": "Restricting the provider to using data only to deliver the service, or forbidding model training."},
    {"id": "breach_notice", "title": "Slow or vague breach notification",
     "query": "notify of a security incident or personal data breach within hours",
     "ask": "Is breach notice slower than 72 hours, or 'without undue delay' with no fixed deadline at all?",
     "standard": "A fixed deadline of 72 hours or less (e.g. 24 or 48 hours) is standard, even if worded "
                 "'without undue delay and in any event within ...'."},
    {"id": "unilateral_change", "title": "Unilateral price or terms changes",
     "query": "provider may increase fees or modify terms on notice",
     "ask": "Can one party change prices or terms without consent and without a cap?",
     "standard": "Increases capped at a fixed percentage (around 3-7%) or CPI, with advance notice."},
]
CHECK_TITLES = {c["id"]: c["title"] for c in CHECKS}

SYSTEM = """You review commercial contracts for an in-house legal team and flag clauses a lawyer should look at.
Judge only the clauses provided, labelled [A], [B], ... Do not assume terms that aren't written.
Assess each clause on its own. A clause "creates_risk" only if its wording matches the check AND is worse than
the standard position described. A clause that prohibits, caps or limits the risk protects the parties: it does
not create risk. A clause about a different topic does not create risk.
Severity: "high" if it could expose a party to large or open-ended loss, "medium" if it deserves negotiation,
"low" if worth noting only.
Reply with JSON only, in this shape:
{"clauses": [{"label": "A", "says": "what the clause provides, in a few words", "creates_risk": true or false,
  "severity": "high" | "medium" | "low", "reason": "why it is worse than standard, quoting the key words"}]}"""

_lock = threading.Lock()


# --- flag store -------------------------------------------------------------

def load_flags() -> list[dict]:
    if not FLAGS_PATH.exists():
        return []
    return json.loads(FLAGS_PATH.read_text(encoding="utf-8"))


def save_flags(flags: list[dict]) -> None:
    tmp = FLAGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(flags, indent=2), encoding="utf-8")
    tmp.replace(FLAGS_PATH)


def decide(flag_id: str, status: str, reviewer: str = "", note: str = "") -> dict:
    """Record a lawyer's decision on a flag: 'confirmed', 'dismissed', or back to 'open'."""
    with _lock:
        flags = load_flags()
        for f in flags:
            if f["id"] == flag_id:
                f.update(status=status, reviewer=reviewer.strip(), note=note.strip(),
                         reviewed_at=_now() if status != "open" else None)
                save_flags(flags)
                return f
    raise KeyError(flag_id)


def replace_open_flags(doc_id: str, new: list[dict]) -> None:
    """Swap a re-reviewed agreement's open flags for `new`, keeping lawyers' decisions.

    A new flag on the same clause and check as an already-decided one is skipped,
    so a dismissed flag doesn't come back on every review.
    """
    with _lock:
        flags = load_flags()
        decided = [f for f in flags if f["doc_id"] == doc_id and f["status"] != "open"]
        seen = {(f["check"], f["clause_id"]) for f in decided}
        keep = [f for f in flags if f["doc_id"] != doc_id or f["status"] != "open"]
        save_flags(keep + [f for f in new if (f["check"], f["clause_id"]) not in seen])


def drop_flags(doc_id: str) -> None:
    with _lock:
        save_flags([f for f in load_flags() if f["doc_id"] != doc_id])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- review -----------------------------------------------------------------

def candidates(chunks: list[Chunk], embeddings: np.ndarray, doc_id: str, query: str, gate: bool = True) -> list[Chunk]:
    idx = [i for i, c in enumerate(chunks) if c.doc_id == doc_id]
    scores = embeddings[idx] @ embed_query(query)
    return [chunks[idx[j]] for j in np.argsort(-scores)[:CLAUSES_PER_CHECK]
            if not gate or scores[j] >= MIN_CHECK_SIMILARITY]


def run_check(check: dict, clauses: list[Chunk], verify: bool = True) -> dict | None:
    """Ask the model about one check; return a flag dict, or None if nothing is flagged.
    `verify=False` skips the second call (used by the evaluation to measure what it adds)."""
    import ollama

    if not clauses:
        return None

    context = "\n\n".join(f"[{chr(65 + i)}] Section {c.section}\n{c.text}" for i, c in enumerate(clauses))
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Agreement: {clauses[0].doc_title}\n\nClauses:\n\n{context}\n\n"
                                        f"Check: {check['title']}. {check['ask']}\n"
                                        f"Standard (not a risk): {check['standard']}"},
        ],
        format="json",
        options={"temperature": 0},
    )
    try:
        assessed = json.loads(response.message.content).get("clauses", [])
    except (json.JSONDecodeError, AttributeError):
        return None

    # Keep the worst risky clause. Labels are range-checked: a flag must point at
    # a clause the model was shown.
    risky = []
    for a in assessed if isinstance(assessed, list) else []:
        if not isinstance(a, dict) or a.get("creates_risk") is not True:
            continue
        label = str(a.get("label", "")).strip().strip("[]").upper()[:1]
        idx = ord(label) - 65 if label else -1
        severity = str(a.get("severity", "")).lower()
        if 0 <= idx < len(clauses):
            risky.append((idx, severity if severity in SEVERITIES else "medium", str(a.get("reason", "")).strip()))
    risky = [r for r in sorted(risky, key=lambda r: SEVERITIES.index(r[1]))
             if not verify or confirm(check, clauses[r[0]])]
    if not risky:
        return None
    idx, severity, reason = risky[0]
    clause = clauses[idx]
    return {
        "id": uuid.uuid4().hex[:12],
        "doc_id": clause.doc_id,
        "doc_title": clause.doc_title,
        "check": check["id"],
        "check_title": check["title"],
        "clause_id": clause.id,
        "clause": clause.clause,
        "section": clause.section,
        "severity": severity,
        "reason": reason,
        "model": MODEL,
        "status": "open",
        "reviewer": "",
        "note": "",
        "created_at": _now(),
        "reviewed_at": None,
    }


VERIFY = """You double-check a junior reviewer's flag on one contract clause.
Compare the clause with the standard position. Reply with JSON only:
{"says": "what the clause provides, in a few words", "about_this_risk": true or false, "worse_than_standard": true or false}
Answer true only if the clause is clearly worse for a party than the standard position. A clause that matches,
or is better than, the standard is false. "about_this_risk" is false if the clause deals with a different topic
than the risk checked (e.g. a liability cap when the risk is price changes)."""


def confirm(check: dict, clause: Chunk) -> bool:
    """Second, narrower model call on one proposed flag. The first pass sees several
    clauses and over-flags; asked about one clause against the standard, the
    model drops most of those false positives."""
    import ollama

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": VERIFY},
            {"role": "user", "content": f"Clause ({clause.doc_title}, Section {clause.section}):\n{clause.text}\n\n"
                                        f"Risk checked: {check['ask']}\nStandard position: {check['standard']}"},
        ],
        format="json",
        options={"temperature": 0},
    )
    try:
        verdict = json.loads(response.message.content)
        return verdict.get("about_this_risk") is True and verdict.get("worse_than_standard") is True
    except (json.JSONDecodeError, AttributeError):
        return False


def find_flags(chunks: list[Chunk], embeddings: np.ndarray, doc_id: str, on_progress=None,
               gate: bool = True, verify: bool = True) -> list[dict]:
    """Run every check on one agreement and return its flags, without storing them."""
    flags = []
    for i, check in enumerate(CHECKS):
        if on_progress:
            on_progress(i, len(CHECKS))
        flag = run_check(check, candidates(chunks, embeddings, doc_id, check["query"], gate), verify)
        if flag and flag["clause_id"] not in {f["clause_id"] for f in flags}:  # one flag per clause
            flags.append(flag)
    return flags


def review_document(chunks: list[Chunk], embeddings: np.ndarray, doc_id: str, on_progress=None) -> list[dict]:
    """Run every check on one agreement, store its flags for review, and return them."""
    flags = find_flags(chunks, embeddings, doc_id, on_progress)
    replace_open_flags(doc_id, flags)
    return flags


if __name__ == "__main__":
    import sys

    from .index import load_index

    chunks, embeddings = load_index()
    doc_ids = sys.argv[1:] or list(dict.fromkeys(c.doc_id for c in chunks))
    for doc_id in doc_ids:
        print(f"\n{doc_id}")
        flags = review_document(chunks, embeddings, doc_id,
                                on_progress=lambda i, n: print(f"  check {i + 1}/{n}", end="\r"))
        for f in flags:
            print(f"  [{f['severity']:6s}] §{f['clause']:5s} {f['check_title']}: {f['reason']}")
        if not flags:
            print("  no flags")
