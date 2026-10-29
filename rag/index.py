"""Build and load the on-disk index: chunk metadata + dense embeddings.

BM25 is rebuilt in memory at load time (it takes milliseconds for a corpus this
size), so only the embeddings need to be persisted.
"""

import json
from pathlib import Path

import numpy as np

from .chunk import Chunk, load_corpus

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "contracts"
INDEX_DIR = ROOT / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_embedder = None


def get_embedder():
    # Lazy: fastembed downloads the ONNX model (~130 MB) on first use.
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


def embed_passages(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(get_embedder().passage_embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_query(text: str) -> np.ndarray:
    # bge models expect an instruction prefix on queries; query_embed adds it.
    vec = np.array(next(iter(get_embedder().query_embed([text]))), dtype=np.float32)
    return vec / np.linalg.norm(vec)


def build_index(corpus_dir: Path = CORPUS_DIR, index_dir: Path = INDEX_DIR) -> list[Chunk]:
    chunks = load_corpus(corpus_dir)
    embeddings = embed_passages([c.indexed_text for c in chunks])
    index_dir.mkdir(exist_ok=True)
    (index_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2), encoding="utf-8"
    )
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "meta.json").write_text(
        json.dumps({"embed_model": EMBED_MODEL, "n_chunks": len(chunks)}), encoding="utf-8"
    )
    return chunks


def load_index(index_dir: Path = INDEX_DIR) -> tuple[list[Chunk], np.ndarray]:
    if not (index_dir / "chunks.json").exists():
        raise SystemExit("No index found. Run `python -m rag.build` first.")
    chunks = [Chunk(**d) for d in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
    return chunks, np.load(index_dir / "embeddings.npy")
