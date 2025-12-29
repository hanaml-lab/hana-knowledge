"""Storage interfaces, with in-memory implementations used by default and in tests."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from graph_memory.model import Edge, Entity


class GraphStore(Protocol):
    def upsert_entity(self, entity: Entity) -> None: ...
    def entity(self, entity_id: str) -> Entity | None: ...
    def entities(self) -> list[Entity]: ...
    def add_edge(self, edge: Edge) -> None: ...
    def retract_edge(self, edge: Edge, at) -> None: ...
    def edges(self, subject: str | None = None, relation: str | None = None,
              include_retracted: bool = False) -> list[Edge]: ...


class VectorStore(Protocol):
    def upsert(self, key: str, vector: np.ndarray, payload: dict) -> None: ...
    def search(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float, dict]]: ...


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._edges: dict[int, Edge] = {}

    def upsert_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity

    def entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def entities(self) -> list[Entity]:
        return list(self._entities.values())

    def add_edge(self, edge: Edge) -> None:
        self._edges[edge.id] = edge

    def retract_edge(self, edge: Edge, at) -> None:
        self._edges[edge.id] = Edge(**{**edge.__dict__, "retracted_at": at})

    def edges(self, subject=None, relation=None, include_retracted=False) -> list[Edge]:
        return [e for e in self._edges.values()
                if (subject is None or e.subject == subject)
                and (relation is None or e.relation == relation)
                and (include_retracted or e.current)]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._keys: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._payloads: list[dict] = []

    def upsert(self, key: str, vector: np.ndarray, payload: dict) -> None:
        if key in self._keys:
            i = self._keys.index(key)
            self._vectors[i], self._payloads[i] = vector, payload
        else:
            self._keys.append(key)
            self._vectors.append(vector)
            self._payloads.append(payload)

    def search(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float, dict]]:
        if not self._keys:
            return []
        scores = np.stack(self._vectors) @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._keys[i], float(scores[i]), self._payloads[i]) for i in order]
