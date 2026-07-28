"""
Embeddings + top-m cosine retrieval (Eq. 7).

Three backends, chosen by memory.embedding_backend:

  hash                  dependency-free hashed bag of character n-grams and
                        words. Deterministic, no download, no GPU. The default,
                        so a run never blocks on fetching an embedding model.
  sentence_transformers a real sentence encoder, if the package is installed.
  backbone              mean-pooled hidden states from the same LLM the rest of
                        the framework uses.

All backends return L2-normalized rows, so cosine similarity is a dot product.
"""

import hashlib
import re
from typing import List, Optional, Sequence

import numpy as np

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\sA-Za-z_0-9]")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


class HashEmbedder:
    """Hashed bag of words + character 4-grams. Deterministic across runs."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = (text or "").lower()
            for tok in _TOKEN.findall(lowered):
                out[row, self._bucket(tok)] += 1.0
            for i in range(len(lowered) - 3):
                out[row, self._bucket(lowered[i:i + 4])] += 0.5
        # Damp frequent tokens before normalizing.
        return _l2_normalize(np.log1p(out))


class SentenceTransformerEmbedder:
    name = "sentence_transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), convert_to_numpy=True,
                                 show_progress_bar=False)
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))


class BackboneEmbedder:
    """Mean-pooled last hidden states of the LLM already in memory."""

    name = "backbone"

    def __init__(self, backbone, max_tokens: int = 512):
        self.backbone = backbone
        self.max_tokens = int(max_tokens)
        self.dim = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        tok = self.backbone.tokenizer
        model = self.backbone.model
        rows = []
        with torch.no_grad():
            for text in texts:
                enc = tok(text or "", return_tensors="pt", truncation=True,
                          max_length=self.max_tokens).to(model.device)
                out = model(**enc, output_hidden_states=True)
                hidden = out.hidden_states[-1][0]                  # (T, H)
                mask = enc["attention_mask"][0].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(0) / mask.sum().clamp(min=1)
                rows.append(pooled.float().cpu().numpy())
        mat = np.stack(rows).astype(np.float32)
        self.dim = mat.shape[1]
        return _l2_normalize(mat)


def build_embedder(cfg, backbone=None):
    """cfg is a MemoryConfig. Falls back to the hash embedder if a dep is missing."""
    backend = (cfg.embedding_backend or "hash").lower()
    if backend == "sentence_transformers":
        try:
            return SentenceTransformerEmbedder(cfg.embedding_model)
        except Exception as e:                       # not installed / no network
            print(f"[memory] sentence_transformers unavailable ({e}); using hash")
            return HashEmbedder(cfg.embedding_dim)
    if backend == "backbone":
        if backbone is None:
            print("[memory] no backbone passed for embedding; using hash")
            return HashEmbedder(cfg.embedding_dim)
        return BackboneEmbedder(backbone)
    return HashEmbedder(cfg.embedding_dim)


def top_m_by_cosine(query: np.ndarray, matrix: np.ndarray, m: int) -> List[int]:
    """Indices of the m most similar rows. Inputs are assumed L2-normalized."""
    if matrix.size == 0 or m <= 0:
        return []
    sims = matrix @ query
    m = min(int(m), sims.shape[0])
    idx = np.argpartition(-sims, m - 1)[:m]
    return [int(i) for i in idx[np.argsort(-sims[idx])]]
