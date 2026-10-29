"""Checks that keep answers tied to the agreements, enforced in code rather than
left to the prompt.

1. Relevance gate: if no clause is close in meaning to the question, the model
   is never called and the question is declined. The threshold sits in the gap
   between the weakest real contract question (cosine 0.64) and the strongest
   off-topic one (0.54, "Explain quantum computing") measured on this corpus.
2. Citation coverage: every sentence of an answer should carry a clause label.
   Sentences without one are returned so the caller can flag them.
"""

import re

MIN_SIMILARITY = 0.60
REFUSAL = "The provided clauses don't address this."

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LABEL = re.compile(r"\[[A-Z]\]")
# Company names and legal shorthand end in a full stop without ending the sentence.
_ABBREV = re.compile(r"\b(?:Inc|Corp|Co|Ltd|LLC|No|Sec|vs|e\.g|i\.e|U\.S)\.$", re.IGNORECASE)


def is_relevant(top_similarity: float) -> bool:
    return top_similarity >= MIN_SIMILARITY


def uncited_sentences(text: str) -> list[str]:
    """Sentences in a raw [A]-labelled answer that cite no clause.

    Labels sometimes land after the full stop ("...45 days. [B]"), so a piece
    made only of labels is attached to the sentence before it, as is anything
    following an abbreviation like "Inc.".
    """
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text.strip()):
        if sentences and (not _LABEL.sub("", piece).strip() or _ABBREV.search(sentences[-1])):
            sentences[-1] += " " + piece
        elif piece.strip():
            sentences.append(piece.strip())
    return [s for s in sentences if not _LABEL.search(s) and "don't address" not in s.replace("’", "'")]
