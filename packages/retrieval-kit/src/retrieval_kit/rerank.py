"""Rerankers score (question, passage) pairs jointly."""

from __future__ import annotations

import math
from typing import Protocol

from retrieval_kit.index.bm25 import tokenize


class Reranker(Protocol):
    def score(self, question: str, passages: list[str]) -> list[float]: ...


class OverlapReranker:
    """Weighted term overlap squashed to [0, 1]. A dependency-free stand-in for a cross-encoder."""

    def score(self, question: str, passages: list[str]) -> list[float]:
        q = set(tokenize(question))
        if not q:
            return [0.0] * len(passages)
        out = []
        for p in passages:
            terms = set(tokenize(p))
            hit = sum(1.0 + math.log(len(t)) for t in q & terms)
            total = sum(1.0 + math.log(len(t)) for t in q)
            out.append(hit / total)
        return out


class CrossEncoderReranker:
    """bge-reranker-v2-m3 or hanaml/rerank-m through sentence-transformers."""

    def __init__(self, name: str, batch_size: int = 16):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("cross-encoders need: pip install 'retrieval-kit[models]'") from exc
        self.model = CrossEncoder(name, activation_fn=None)
        self.batch_size = batch_size

    def score(self, question: str, passages: list[str]) -> list[float]:  # pragma: no cover
        logits = self.model.predict([(question, p) for p in passages], batch_size=self.batch_size)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def load_reranker(name: str | None) -> Reranker:
    if name in (None, "", "overlap"):
        return OverlapReranker()
    return CrossEncoderReranker(name)
