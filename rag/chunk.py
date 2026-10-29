"""Clause-level chunking for contracts.

Legal text has natural boundaries (numbered clauses), so instead of fixed-size
windows we split on them. Each chunk is prefixed with its contract title and
section heading ("contextual chunk header") so a clause like "7.2 ... SHALL NOT
EXCEED THE FEES PAID ..." still carries which agreement and topic it belongs to.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path

CLAUSE_RE = re.compile(r"^(\d+\.\d+)\s+", re.MULTILINE)


@dataclass
class Chunk:
    id: str          # e.g. "msa_northwind_quillon#7.2"
    doc_id: str
    doc_title: str
    section: str     # e.g. "7. Limitation of Liability"
    clause: str      # e.g. "7.2"
    text: str        # clause text only
    indexed_text: str  # header + clause text; what gets embedded / BM25-indexed

    def to_dict(self):
        return asdict(self)


def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    doc_id = path.stem
    title_match = re.search(r"^# (.+)$", raw, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else doc_id

    chunks = []
    # Split into "## <section>" blocks; the first piece is the preamble.
    for block in re.split(r"^## ", raw, flags=re.MULTILINE)[1:]:
        heading, _, body = block.partition("\n")
        heading = heading.strip()
        starts = list(CLAUSE_RE.finditer(body))
        for i, m in enumerate(starts):
            end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
            clause_text = body[m.start():end].strip()
            clause = m.group(1)
            chunks.append(Chunk(
                id=f"{doc_id}#{clause}",
                doc_id=doc_id,
                doc_title=doc_title,
                section=heading,
                clause=clause,
                text=clause_text,
                indexed_text=f"{doc_title}\nSection {heading}\n{clause_text}",
            ))
    return chunks


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_document(path))
    return chunks
