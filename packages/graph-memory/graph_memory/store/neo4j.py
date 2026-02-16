"""Neo4j 5 graph store. Edges are relationships of type FACT carrying both time axes.

Requires ``pip install "graph-memory[neo4j]"``; the schema is in ``cypher/schema.cypher``.
"""

from __future__ import annotations

from datetime import date

from graph_memory.model import Edge, Entity

_EDGE_FIELDS = "f.id AS id, s.id AS subject, f.relation AS relation, o.id AS object, " \
    "f.valid_from AS valid_from, f.valid_to AS valid_to, f.recorded_at AS recorded_at, " \
    "f.retracted_at AS retracted_at, f.source AS source"


def _d(value) -> date | None:
    if value is None:
        return None
    return value.to_native() if hasattr(value, "to_native") else value


class Neo4jStore:
    def __init__(self, uri: str, user: str = "neo4j", password: str = "graph-memory", database: str = "neo4j"):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError('Neo4j support needs: pip install "graph-memory[neo4j]"') from exc
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, **params):
        records, _, _ = self.driver.execute_query(query, params, database_=self.database)
        return records

    def upsert_entity(self, entity: Entity) -> None:
        self._run("MERGE (e:Entity {id: $id}) SET e.name = $name, e.kind = $kind",
                  id=entity.id, name=entity.name, kind=entity.kind)

    def entity(self, entity_id: str) -> Entity | None:
        rows = self._run("MATCH (e:Entity {id: $id}) RETURN e.id AS id, e.name AS name, e.kind AS kind",
                         id=entity_id)
        return Entity(rows[0]["id"], rows[0]["name"], rows[0]["kind"]) if rows else None

    def entities(self) -> list[Entity]:
        rows = self._run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.kind AS kind")
        return [Entity(r["id"], r["name"], r["kind"]) for r in rows]

    def add_edge(self, edge: Edge) -> None:
        self._run(
            "MATCH (s:Entity {id: $subject}), (o:Entity {id: $object}) "
            "CREATE (s)-[:FACT {id: $id, relation: $relation, valid_from: $valid_from, valid_to: $valid_to, "
            "recorded_at: $recorded_at, retracted_at: $retracted_at, source: $source}]->(o)",
            **edge.__dict__,
        )

    def retract_edge(self, edge: Edge, at: date) -> None:
        self._run("MATCH ()-[f:FACT {id: $id}]->() SET f.retracted_at = $at", id=edge.id, at=at)

    def edges(self, subject=None, relation=None, include_retracted=False) -> list[Edge]:
        rows = self._run(
            "MATCH (s:Entity)-[f:FACT]->(o:Entity) "
            "WHERE ($subject IS NULL OR s.id = $subject) AND ($relation IS NULL OR f.relation = $relation) "
            "AND ($all OR f.retracted_at IS NULL) RETURN " + _EDGE_FIELDS,
            subject=subject, relation=relation, all=include_retracted,
        )
        return [Edge(subject=r["subject"], relation=r["relation"], object=r["object"],
                     valid_from=_d(r["valid_from"]), valid_to=_d(r["valid_to"]),
                     recorded_at=_d(r["recorded_at"]), retracted_at=_d(r["retracted_at"]),
                     source=r["source"] or "", id=r["id"]) for r in rows]
