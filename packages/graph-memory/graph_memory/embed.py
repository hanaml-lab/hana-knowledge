"""Embeddings for entity names and summaries."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class CharNgramEmbedder:
    """Hashed character trigrams. Robust to spelling variants; needs no weights."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text.lower().strip()}  "
            for i in range(len(padded) - 2):
                h = int.from_bytes(hashlib.blake2b(padded[i:i + 3].encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class BgeM3Embedder:
    def __init__(self, name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
