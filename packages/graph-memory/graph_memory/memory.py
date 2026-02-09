"""The public entry point."""

from __future__ import annotations

from datetime import date

from graph_memory import communities as comm
from graph_memory import notes
from graph_memory.embed import BgeM3Embedder, CharNgramEmbedder, Embedder
from graph_memory.extract import Extractor, LLMExtractor, PatternExtractor
from graph_memory.model import Edge, Entity
from graph_memory.recall import Recall, pack
from graph_memory.resolve import Resolver
from graph_memory.store import GraphStore, InMemoryGraphStore, InMemoryVectorStore, VectorStore


def _date(value: str | date | None) -> date:
    if value is None:
        return date.today()
    return value if isinstance(value, date) else date.fromisoformat(value)


class Memory:
    """``Memory()`` keeps everything in process. Pass ``neo4j=`` and ``qdrant=`` URLs to persist."""

    def __init__(
        self,
        neo4j: str | None = None,
        qdrant: str | None = None,
        extractor: str | Extractor = "patterns",
        embedder: str | Embedder = "char-ngram",
        resolve_threshold: float = 0.82,
        community_refresh: str = "on_change",
        recall_tokens: int = 4096,
        graph: GraphStore | None = None,
        vectors: VectorStore | None = None,
    ):
        if graph is None and neo4j:
            from graph_memory.store.neo4j import Neo4jStore
            graph = Neo4jStore(neo4j)
        if vectors is None and qdrant:
            from graph_memory.store.qdrant import QdrantStore
            vectors = QdrantStore(qdrant)
        self.graph = graph or InMemoryGraphStore()
        self.vectors = vectors or InMemoryVectorStore()
        if isinstance(extractor, str):
            extractor = PatternExtractor() if extractor == "patterns" else LLMExtractor(extractor)
        self.extractor = extractor
        if isinstance(embedder, str):
            embedder = CharNgramEmbedder() if embedder == "char-ngram" else BgeM3Embedder(embedder)
        self.resolver = Resolver(self.graph, self.vectors, embedder, resolve_threshold)
        if community_refresh not in {"on_change", "nightly"}:
            raise ValueError("community_refresh is on_change or nightly")
        self.community_refresh = community_refresh
        self.recall_tokens = recall_tokens
        self._communities: list[comm.Community] = []
        self._stale = False

    def add(self, user: str, text: str, at: str | date | None = None) -> list[Edge]:
        when = _date(at)
        if self.graph.entity(user) is None:
            self.graph.upsert_entity(Entity(user, user, "person"))
        changed: list[Edge] = []
        for item in self.extractor.extract(user, text, when):
            changed += self.resolver.apply(item, user, when, source=text)
        if changed:
            self._stale = True
            if self.community_refresh == "on_change":
                self.refresh_communities(when)
        return changed

    def refresh_communities(self, at: str | date | None = None) -> None:
        self._communities = comm.detect(self.graph, _date(at))
        self._stale = False

    def recall(self, user: str, query: str, at: str | date | None = None) -> Recall:
        now = _date(at)
        if self._stale and self.community_refresh == "on_change":
            self.refresh_communities(now)
        names = {e.id: e.name for e in self.graph.entities()}
        mine = [c for c in self._communities if user in c.members]
        return pack(self.graph.edges(subject=user), mine, query, now, names, self.recall_tokens)

    def export_notes(self, user: str | None = None) -> str:
        names = {e.id: e.name for e in self.graph.entities()}
        return notes.dump(self.graph.edges(subject=user, include_retracted=True), names)
