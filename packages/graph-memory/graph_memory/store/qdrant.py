"""Qdrant vector store for entity and summary embeddings.

Requires ``pip install "graph-memory[qdrant]"``.
"""

from __future__ import annotations

import uuid

import numpy as np


class QdrantStore:
    def __init__(self, url: str, collection: str = "graph_memory", dim: int = 1024):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ImportError('Qdrant support needs: pip install "graph-memory[qdrant]"') from exc
        self.models = models
        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def upsert(self, key: str, vector: np.ndarray, payload: dict) -> None:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
        self.client.upsert(self.collection, points=[
            self.models.PointStruct(id=point_id, vector=vector.tolist(), payload={**payload, "key": key})])

    def search(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float, dict]]:
        hits = self.client.query_points(self.collection, query=vector.tolist(), limit=k).points
        return [(h.payload["key"], float(h.score), h.payload) for h in hits]
