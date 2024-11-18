"""Dense retrieval: an encoder plus exact inner-product search over normalised vectors."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from retrieval_kit.index.bm25 import tokenize


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEncoder:
    """Feature-hashed unigrams and bigrams. Needs no model weights; used offline and in tests."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            for feat in toks + [a + " " + b for a, b in zip(toks, toks[1:])]:
                col, sign = self._bucket(feat)
                out[row, col] += sign
        return _normalise(out)


class SentenceTransformerEncoder:
    """bge-m3 or any sentence-transformers model, encoded in batches."""

    def __init__(self, name: str, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("dense models need: pip install 'retrieval-kit[models]'") from exc
        self.model = SentenceTransformer(name)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs weights
        vecs = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
        return _normalise(vecs.astype(np.float32))


def load_encoder(name: str | None) -> Encoder:
    if name in (None, "", "hashing"):
        return HashingEncoder()
    return SentenceTransformerEncoder(name)


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class DenseIndex:
    def __init__(self, encoder: Encoder, docs: list[str]):
        self.encoder = encoder
        self.vectors = encoder.encode(docs) if docs else np.zeros((0, 1), dtype=np.float32)

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        if len(self.vectors) == 0:
            return []
        q = self.encoder.encode([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]
