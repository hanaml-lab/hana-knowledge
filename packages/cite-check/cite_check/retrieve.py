"""Find, inside each cited source, the passage most likely to support a claim.

Sources are chunked and searched with retrieval-kit (BM25 + dense, fused, then reranked), the
same retrieval the answering side of hana-knowledge uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from retrieval_kit import Index, Retriever
from retrieval_kit.index import Chunk


@dataclass(frozen=True)
class Evidence:
    source: str
    page: int
    text: str
    score: float


class SourceLibrary:
    def __init__(self, folder: str | Path, chunk_tokens: int = 96, overlap: int = 24,
                 reranker: str | None = None):
        self.folder = Path(folder)
        self.index = Index.build(self.folder, chunk_tokens=chunk_tokens, overlap=overlap)
        self.retriever = Retriever(self.index, reranker=reranker, k=5)
        self.tables = {p.stem.lower(): p for p in self.folder.glob("*.csv")}

    def _matches(self, chunk: Chunk, source: str, page: int | None) -> bool:
        name = source.lower()
        doc = chunk.doc.lower()
        return (doc == name or Path(doc).stem == name) and (page is None or chunk.page == page)

    def passages(self, claim: str, source: str, page: int | None = None, k: int = 3) -> list[Evidence]:
        hits = [h for h in self.retriever.search(claim, k=50) if self._matches(h.chunk, source, page)]
        if not hits:  # a cited page with no lexical overlap still deserves a look
            hits_chunks = [c for c in self.index.chunks if self._matches(c, source, page)][:k]
            return [Evidence(c.doc, c.page, c.text, 0.0) for c in hits_chunks]
        return [Evidence(h.chunk.doc, h.chunk.page, h.chunk.text, h.score) for h in hits[:k]]

    def table(self, source: str) -> Path | None:
        return self.tables.get(Path(source).stem.lower())

    def has(self, source: str) -> bool:
        return any(self._matches(c, source, None) for c in self.index.chunks) or self.table(source) is not None
