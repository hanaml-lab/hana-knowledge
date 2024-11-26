"""BM25 + dense retrieval, fused and reranked."""

from __future__ import annotations

from dataclasses import dataclass

from retrieval_kit.fuse import rrf
from retrieval_kit.index import Chunk, Index, load_encoder
from retrieval_kit.index.dense import DenseIndex
from retrieval_kit.rerank import Reranker, load_reranker


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # reranker score in [0, 1]


class Retriever:
    def __init__(
        self,
        index: Index,
        dense: str | None = None,
        reranker: str | Reranker | None = None,
        k: int = 10,
        candidates: int = 100,
        rrf_k: int = 60,
        rerank_top: int = 50,
    ):
        self.index = index
        if dense is not None:
            index.encoder = load_encoder(dense)
            index.dense = DenseIndex(index.encoder, [c.text for c in index.chunks])
        self.reranker = reranker if not isinstance(reranker, str | None) else load_reranker(reranker)
        self.k, self.candidates, self.rrf_k, self.rerank_top = k, candidates, rrf_k, rerank_top

    def fused(self, query: str) -> list[int]:
        sparse = [i for i, _ in self.index.bm25.search(query, self.candidates)]
        dense = [i for i, _ in self.index.dense.search(query, self.candidates)]
        return [i for i, _ in rrf([sparse, dense], k=self.rrf_k)]

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        top = self.fused(query)[: self.rerank_top]
        chunks = [self.index.chunks[i] for i in top]
        scores = self.reranker.score(query, [c.text for c in chunks])
        hits = sorted((Hit(c, s) for c, s in zip(chunks, scores)), key=lambda h: (-h.score, h.chunk.id))
        return hits[: k or self.k]
