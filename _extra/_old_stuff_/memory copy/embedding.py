"""
The embedding function e(.) behind Eq. 7.

Two backends:

  hash                   deterministic hashed bag of words (unigrams +
                         bigrams) into a fixed-width vector. No extra
                         dependency, no model load, no GPU, and stable across
                         processes -- Python's builtin hash() is salted per
                         process, so crc32 is used instead. Retrieval quality
                         is lexical, which is enough when the corpus is a few
                         hundred lessons about one problem.

  sentence_transformers  a real sentence encoder if the package is installed.
                         Loaded on CPU by default so it does not compete with
                         training or generation for VRAM.

`auto` tries sentence_transformers and falls back to hash, printing which one
it ended up with. Whichever backend is used, encode() returns L2-normalized
rows so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import re
import zlib
from typing import List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Embedder:
    def __init__(self, backend: str = "auto",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dim: int = 2048, device: str = "cpu", verbose: bool = True):
        self.requested = backend
        self.model_name = model_name
        self.dim = int(dim)
        self.device = device
        self._model = None
        self.backend = "hash"

        if backend in ("auto", "sentence_transformers"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name, device=device)
                self.backend = "sentence_transformers"
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception as e:
                if backend == "sentence_transformers":
                    raise RuntimeError(
                        f"memory_embed_backend=sentence_transformers but the "
                        f"model could not be loaded: {e!r}") from e
                if verbose:
                    print(f"[memory] sentence-transformers unavailable ({e!r}); "
                          f"using the hashed bag-of-words embedder")

        if verbose:
            print(f"[memory] embedder = {self.backend} (dim={self.dim})")

    # ------------------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Returns (n, dim), L2-normalized."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "sentence_transformers":
            vecs = self._model.encode(texts, convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        return np.stack([self._hash_vector(t) for t in texts]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    def _hash_vector(self, text: str) -> np.ndarray:
        toks = _tokens(text)
        vec = np.zeros(self.dim, dtype=np.float64)
        if not toks:
            return vec.astype(np.float32)

        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for a, b in zip(toks, toks[1:]):
            bg = a + "_" + b
            counts[bg] = counts.get(bg, 0) + 1

        for term, c in counts.items():
            idx = zlib.crc32(term.encode("utf-8")) % self.dim
            # Signed hashing keeps unrelated collisions from always adding up.
            sign = 1.0 if (zlib.crc32(b"s" + term.encode("utf-8")) & 1) else -1.0
            # Sublinear tf: a term repeated 50 times is not 50x more telling.
            vec[idx] += sign * (1.0 + np.log(c))

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    query: (d,), matrix: (n, d), both already L2-normalized. Returns (n,).
    Renormalizes defensively so a hand-edited memory.json cannot skew ranking.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn > 0:
        q = q / qn
    m = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms) @ q
