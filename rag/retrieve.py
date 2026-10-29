"""Three retrievers over the same chunks: BM25 (lexical), dense (embeddings),
and hybrid (Reciprocal Rank Fusion of the two).

Lexical search wins on exact terms ("SOC 2", "AES-256", party names); dense
search wins on paraphrase ("how fast must they tell us about a hack?" ->
"Personal Data Breach ... within forty-eight (48) hours"). RRF combines ranks,
not scores, so the two don't need calibrating against each other.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk
from .index import embed_query, load_index

RRF_K = 60  # standard constant from Cormack et al. (2009)
METHODS = ("bm25", "dense", "hybrid")

# Question words ("how", "does", "what") otherwise match boilerplate across
# every contract and swamp BM25's signal on short queries.
STOPWORDS = set("""a an and are as at be by can do does for from has have how if in is it its
of on or that the their them they this to under was what when which who will with within
you your we us our must""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, embeddings: np.ndarray | None = None):
        if chunks is None:
            chunks, embeddings = load_index()
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores))

    def _dense_ranking(self, query: str) -> list[int]:
        scores = self.embeddings @ embed_query(query)  # cosine; vectors are normalized
        return list(np.argsort(-scores))

    def _hybrid_ranking(self, query: str) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (self._bm25_ranking(query), self._dense_ranking(query)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)

    def search(self, query: str, k: int = 5, method: str = "dense") -> list[Chunk]:
        ranking = {
            "bm25": self._bm25_ranking,
            "dense": self._dense_ranking,
            "hybrid": self._hybrid_ranking,
        }[method](query)
        return [self.chunks[i] for i in ranking[:k]]
