// graph-memory schema for Neo4j 5.
// Entities are nodes; every fact is a FACT relationship with valid time and recorded time.

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE INDEX entity_kind IF NOT EXISTS
FOR (e:Entity) ON (e.kind);

CREATE INDEX fact_id IF NOT EXISTS
FOR ()-[f:FACT]-() ON (f.id);

CREATE INDEX fact_relation IF NOT EXISTS
FOR ()-[f:FACT]-() ON (f.relation);

// Facts the agent currently believes and that held on a given day:
// MATCH (s:Entity {id: $user})-[f:FACT]->(o:Entity)
// WHERE f.retracted_at IS NULL
//   AND (f.valid_from IS NULL OR f.valid_from <= $day)
//   AND (f.valid_to IS NULL OR $day < f.valid_to)
// RETURN f.relation, o.name, f.valid_from, f.valid_to;
