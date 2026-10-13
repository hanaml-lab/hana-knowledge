# graph-memory

Part of [hana-knowledge](../../README.md). The pipeline uses it to resolve "my employer" and similar phrases to the
fact that was valid when the question was asked.

Memory for agents that talk to the same people for months. Facts are extracted from conversations into a knowledge
graph where every edge carries two times: when the fact was true in the world (`valid_from`, `valid_to`) and when
the agent learned it (`recorded_at`). When someone says "I moved to Leeds in March", the old address is closed off
at March, not deleted, so the agent can still answer "where did I live last year?".

## Install

```bash
# from the root of hana-knowledge
pip install "./packages/graph-memory[all]"
docker compose -f packages/graph-memory/docker-compose.yml up -d   # Neo4j 5 and Qdrant, for local use
```

`pip install ./packages/graph-memory` alone gives an in-process store
and a pattern-based extractor with no services and no model, which is enough to try it and is what the tests use.

## Quickstart

```python
from graph_memory import Memory

memory = Memory(neo4j="bolt://localhost:7687", qdrant="http://localhost:6333",
                extractor="meta-llama/Llama-3.1-8B-Instruct")

memory.add(user="u42", text="I moved to Leeds in March. Still at the same firm.", at="2026-04-02")
memory.add(user="u42", text="Actually it was February, not March.", at="2026-04-09")

hits = memory.recall(user="u42", query="Where does the user live, and since when?")
print(hits.context)     # a few hundred tokens, ready for the reader model's prompt
```

The extractor is called through an OpenAI-compatible endpoint, `EXTRACTOR_URL` (default
`http://localhost:30000/v1`, SGLang's port). `Memory()` with no arguments runs the same example in process.

### Valid time and recorded time

After the two calls above the graph holds two versions of the Leeds fact:

| valid_from | valid_to | recorded_at | retracted_at |
| --- | --- | --- | --- |
| 2026-03-01 | | 2026-04-02 | 2026-04-09 |
| 2026-02-01 | | 2026-04-09 | |

"Where does the user live?" reads the current version. "What did we believe on 5 April?" reads the first row,
because on that day it was recorded and not yet retracted.

### Deleting an episode

Corrections retract; deletions remove. `memory.delete(user="u42", text="I moved to Leeds in March.")` deletes
that turn and rebuilds everything derived from it by replaying the remaining turns in order: the Leeds edge,
the edge it closed (Manchester is open again), entities only it mentioned, their vectors and the community
summaries over them. `memory.epoch` counts deletions and every `Recall` carries the epoch it was built in, so a
summary built before a deletion is never served after it. A store that holds facts from turns this `Memory`
never saw (opened from another process) is not rebuilt; `delete` raises instead.

## How it works

- **Extract.** A small model turns each turn into entities and dated relations, constrained to a JSON schema.
- **Resolve.** New entities are matched to existing ones by name, embedding (bge-m3) and neighbourhood. A relation that contradicts an open edge closes that edge rather than adding a second one.
- **Communities.** Leiden clustering groups related entities; each group gets a short summary that is refreshed when its edges change.
- **Recall.** A question retrieves community summaries and edges, filtered to what was valid at the time the question asks about, and packs them into a small context.

The `packages/notes-reader` package (at the monorepo root) reads the plain dated-notes export (`graph_memory.notes`, or
`memory.export_notes()`) from TypeScript, for apps that only need to show what an agent remembers.

## Configuration

| Setting | Default | What it does |
| --- | --- | --- |
| `extractor` | `patterns` | Model for entity and relation extraction, e.g. `meta-llama/Llama-3.1-8B-Instruct` |
| `embedder` | `char-ngram` | Or `BAAI/bge-m3` with the `embed` extra |
| `resolve_threshold` | 0.82 | Similarity above which two entities are merged |
| `community_refresh` | `on_change` | Or `nightly`, to batch summary updates |
| `recall_tokens` | 4096 | Budget for the packed context |

## Benchmarks

LongMemEval-S, with the same reader model in both columns. `python packages/graph-memory/bench/longmemeval.py --split s` (reader and
judge through `OPENAI_BASE_URL`; the dataset is downloaded on first run).

| Metric | Full history in 128k context | graph-memory |
| --- | --- | --- |
| Accuracy | 58.2% | 69.4% |
| Temporal-reasoning subset | 41% | 63% |
| Knowledge-update subset | 52% | 74% |
| Median context tokens per answer | 96k | 4.1k |

On a synthetic set of edits, a vector-store memory returned the stale fact after a correction 23% of the time;
graph-memory did 3% of the time.

## Development

```bash
cd packages/graph-memory && pytest
cd packages/notes-reader && npm test
```

## Licence

MIT. See `LICENSE` in this folder.
