"""Grounded answer generation with a local model served by Ollama.

Retrieved clauses are labelled [A]..[E] in the prompt, the model is told to tag
every sentence with the label of the clause that supports it, and the labels are
parsed and range-checked here: a label that doesn't match a retrieved clause is
dropped, so a citation can only ever point at text the model was actually shown.
"""

import re
from dataclasses import dataclass, field

from .chunk import Chunk
from .grounding import REFUSAL, is_relevant, uncited_sentences

MODEL = "qwen2.5:7b"

SYSTEM = """You answer questions about commercial contracts for an in-house legal team.

Use only the contract clauses provided. Cite the clause for every factual statement.
If the clauses don't answer the question, say "The provided clauses don't address this." and stop --
don't fill gaps from general legal knowledge. Name the agreement you're quoting when more than one is
in play, since different agreements often set different terms for the same topic.
Keep answers short: one to four sentences."""

# Clauses are labelled with letters, not numbers: the clause text already
# contains numbers like "2.1", and numeric labels got confused with them.
CITE_FORMAT = """
The clauses are labelled [A], [B], ... End every sentence with the label of the clause that supports it,
e.g. "Invoices are due within 45 days [B]." Only cite labels that appear below."""

CITE_RE = re.compile(r"\[([A-Z])\]")


@dataclass
class Citation:
    chunk_id: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict | None = None
    # Same answer with each [A]-style label replaced by "[n]", n = 1-based
    # position of the clause in the retrieved list, which is what the UI shows.
    marked_text: str = ""
    top_similarity: float | None = None
    gated: bool = False  # declined by the relevance gate; the model was never called
    uncited: list[str] = field(default_factory=list)  # sentences with no clause label

    @property
    def cited_chunk_ids(self) -> list[str]:
        return list(dict.fromkeys(c.chunk_id for c in self.citations))


def answer(question: str, chunks: list[Chunk], top_similarity: float | None = None) -> Answer:
    """Answer from `chunks` only. Pass the question's `top_similarity` to apply the relevance gate."""
    if top_similarity is not None and not is_relevant(top_similarity):
        return Answer(text=REFUSAL, marked_text=REFUSAL, stop_reason="off_topic",
                      top_similarity=top_similarity, gated=True)

    import ollama

    context = "\n\n".join(
        f"[{chr(65 + i)}] {c.doc_title} -- Section {c.section}\n{c.text}" for i, c in enumerate(chunks)
    )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM + CITE_FORMAT},
            {"role": "user", "content": f"Contract clauses:\n\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    text = response.message.content.strip()

    citations = []
    for label in CITE_RE.findall(text):
        idx = ord(label) - 65
        if idx < len(chunks):  # drop hallucinated labels
            citations.append(Citation(chunk_id=chunks[idx].id, cited_text=chunks[idx].text))

    def to_marker(m: re.Match) -> str:
        idx = ord(m.group(1)) - 65
        return f"[{idx + 1}]" if idx < len(chunks) else ""

    return Answer(
        text=text,
        citations=citations,
        stop_reason=response.done_reason,
        usage={"input_tokens": response.prompt_eval_count, "output_tokens": response.eval_count},
        marked_text=CITE_RE.sub(to_marker, text),
        top_similarity=top_similarity,
        uncited=uncited_sentences(text),
    )
