"""Match extracted names to existing entities and fold new facts into the graph.

Relations in ``FUNCTIONAL`` hold one value at a time (a person lives in one place). A new value
closes the open edge at the new fact's ``valid_from`` rather than sitting beside it.
"""

from __future__ import annotations

import re
from datetime import date

from graph_memory.embed import Embedder
from graph_memory.extract import Correction, Fact
from graph_memory.model import Edge, Entity
from graph_memory.store import GraphStore, VectorStore

FUNCTIONAL = {"lives_in", "works_at", "partner", "job_title", "manager"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class Resolver:
    def __init__(self, graph: GraphStore, vectors: VectorStore, embedder: Embedder, threshold: float = 0.82):
        self.graph, self.vectors, self.embedder, self.threshold = graph, vectors, embedder, threshold

    def entity(self, name: str, kind: str, near: str | None = None) -> Entity:
        """Return the entity ``name`` refers to, creating it if nothing is similar enough.

        Similarity is the embedding cosine of the names, plus a small bonus when the candidate
        is already a neighbour of ``near`` (the fact's subject).
        """
        vec = self.embedder.embed([name])[0]
        neighbours = {e.object for e in self.graph.edges(subject=near)} if near else set()
        best: tuple[float, str] | None = None
        for key, score, payload in self.vectors.search(vec, k=5):
            if payload.get("kind") != kind:
                continue
            score += 0.05 if key in neighbours else 0.0
            if best is None or score > best[0]:
                best = (score, key)
        if best and best[0] >= self.threshold:
            found = self.graph.entity(best[1])
            if found:
                return found
        entity = Entity(f"{kind}:{slug(name)}", name, kind)
        self.graph.upsert_entity(entity)
        self.vectors.upsert(entity.id, vec, {"name": name, "kind": kind})
        return entity

    def apply(self, item: Fact | Correction, user: str, at: date, source: str = "") -> list[Edge]:
        if isinstance(item, Correction):
            return self._correct(user, item, at)
        return self._add(item, at, source)

    def _add(self, fact: Fact, at: date, source: str) -> list[Edge]:
        if fact.keeps_previous:
            return []
        obj = self.entity(fact.object, fact.object_kind, near=fact.subject)
        start = fact.valid_from or at
        changed: list[Edge] = []
        for edge in self.graph.edges(subject=fact.subject, relation=fact.relation):
            if edge.valid_to is not None and edge.valid_to <= start:
                continue  # already closed before the new fact begins
            if edge.object == obj.id:
                return []  # we already know this
            if fact.relation in FUNCTIONAL:
                closed = edge.evolve(valid_to=start, recorded_at=at)
                self.graph.retract_edge(edge, at)
                self.graph.add_edge(closed)
                changed.append(closed)
        new = Edge(fact.subject, fact.relation, obj.id, start, None, at, source=source)
        self.graph.add_edge(new)
        return changed + [new]

    def _correct(self, user: str, c: Correction, at: date) -> list[Edge]:
        """Move a start date the user corrected, and the end date of whatever it replaced."""
        changed = []
        for edge in self.graph.edges(subject=user):
            updates = {}
            if edge.valid_from == c.wrong:
                updates["valid_from"] = c.right
            if edge.valid_to == c.wrong:
                updates["valid_to"] = c.right
            if updates:
                fixed = edge.evolve(recorded_at=at, **updates)
                self.graph.retract_edge(edge, at)
                self.graph.add_edge(fixed)
                changed.append(fixed)
        return changed
