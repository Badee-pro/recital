"""python -m rag.build  — chunk the contracts and write the index."""

from collections import Counter

from .index import EMBED_MODEL, INDEX_DIR, build_index

if __name__ == "__main__":
    chunks = build_index()
    per_doc = Counter(c.doc_id for c in chunks)
    print(f"Indexed {len(chunks)} clauses with {EMBED_MODEL} -> {INDEX_DIR}")
    for doc, n in per_doc.items():
        print(f"  {doc}: {n}")
