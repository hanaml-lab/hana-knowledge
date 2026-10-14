# Changelog

## 0.6.0 (2026-10-14)

- retrieval-kit 0.7.0: chunks follow section headings and keep their section path; `Index.section()`
- retrieval-kit: `open_section` and `follow_reference` in the answer loop; a claim whose quote is not in its
  passage triggers one more search, and claims that still fail are listed in `Answer.unanswered`
- graph-memory 0.5.0: budgeted consolidation after TIDE (`Memory(consolidation="budgeted")`, `consolidate()`,
  `graph_memory.consolidate.plan`)
- graph-memory: `Memory.delete()` removes a turn and rebuilds what was derived from it; `Memory.epoch` and
  `Recall.epoch`; `clear()` on every store
- cite-check 0.5.0: physical units converted through pint before figures are compared
- `examples/budgeted_memory.py`; both examples run in CI and in the integration tests
- README with reproduced and reported results, the project's papers and citation; `CITATION.cff`,
  `CONTRIBUTING.md`; package metadata points at this repository

## 0.5.0 (2026-10-07)

- hana_knowledge.KnowledgeBase ties memory, retrieval and verification together
- end-to-end example

## 0.4.0 (2026-06-02)

- cite-check: verify-and-revise pass and CLI

## 0.3.0 (2026-05-21)

- graph-memory notes export and TypeScript reader
- cite-check claims, units and recomputed figures

## 0.2.0 (2026-03-26)

- graph-memory: bitemporal knowledge-graph memory with in-process, Neo4j and Qdrant stores

## 0.1.0 (2025-11-28)

- retrieval-kit: BM25 and dense retrieval, RRF fusion, reranking, cited answers that abstain
