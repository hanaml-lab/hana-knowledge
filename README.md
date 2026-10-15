# hana-knowledge

[![ci](https://github.com/hanaml-lab/hana-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/hana-knowledge/actions/workflows/ci.yml) [![licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)

hanaML's system repo for grounded answers over documents and long-term memory. A question is answered only from
passages in the documents; every sentence of the answer carries a character-span citation, and the answer is
re-checked claim by claim, for support and for its figures recomputed in code, before it is returned. When no
passage supports an answer, the system abstains. Memory for each user is a bitemporal knowledge graph (when a fact
held, and when the system learned it), so a question is resolved against the facts valid at the time it asks
about; a budgeted scheduler decides which turns earn a graph write, and deleting a turn rebuilds everything derived
from it. It is the system of the research project
[Grounded answers over documents](https://hanaml.com/research/grounded-knowledge) and carries parts of the methods of
that project's three Papers.

| Component | Language | What it does |
| --- | --- | --- |
| [`retrieval-kit`](packages/retrieval-kit) | Python, Rust | Hybrid BM25 (Tantivy) and dense retrieval, reciprocal-rank fusion and cross-encoder reranking over section-aware chunks; an answer loop with `search`, `open_section`, `follow_reference` and `answer` tools that cites a character span for every claim or abstains. |
| [`graph-memory`](packages/graph-memory) | Python | Per-user memory as a bitemporal knowledge graph: extraction, entity resolution, contradiction handling, Louvain/Leiden communities and valid-time recall; budgeted consolidation (TIDE) and episode deletion; in-process, Neo4j and Qdrant stores. |
| [`cite-check`](packages/cite-check) | Python | Splits an answer or report into claims, finds each cited passage, runs an NLI, verifier and reviewer cascade, recomputes figures from cited tables (DuckDB) with units converted through pint, and revises flagged claims. |
| [`hana-knowledge`](packages/hana-knowledge) | Python | The pipeline: memory resolves the question, retrieval answers it with citations, cite-check verifies and revises the answer. |
| [`notes-reader`](packages/notes-reader) | TypeScript | Reads graph-memory's dated-notes export, for apps that show a person what the agent remembers. |

## Install

```bash
git clone https://github.com/hanaml-lab/hana-knowledge && cd hana-knowledge
uv sync                                          # every Python package into one environment (Python 3.10+)
pip install ./packages/retrieval-kit/tantivy-ext # optional: BM25 over Tantivy (needs a Rust toolchain)
```

Without extras, dependency-free stand-ins run everything offline: a hashing encoder, a term-overlap reranker, an
extractive answerer and a lexical support check. The extras bring the models: `retrieval-kit[gpu]` or `[cpu]`
(bge-m3 and the cross-encoder), `retrieval-kit[pdf]` (Docling), `graph-memory[all]` (Neo4j, Qdrant, bge-m3,
Leiden), `cite-check[models,sql]` (NLI and verifier models, DuckDB).

## Quickstart

```bash
uv run python examples/end_to_end.py      # memory, retrieval and verification on examples/filings/
uv run python examples/budgeted_memory.py # budgeted consolidation and a deletion over a fortnight of turns
```

```text
Q (2026-05-01): What was the operating margin of my employer in the half year?
  A: The operating margin was 7.2% in the half year.
     cites harbour-h1-2026.txt p.1 chars 128-175
  verified by cite-check: True
...
Q (today): Who won the cup final in 1966?
  abstained: no passage supports an answer
```

```python
from hana_knowledge import KnowledgeBase

kb = KnowledgeBase("examples/filings")
kb.remember("u42", "I work at Northwind Traders.", at="2026-01-10")
kb.remember("u42", "I joined Harbour Logistics in March.", at="2026-04-02")

answer = kb.ask("What was the operating margin of my employer?", user="u42", at="2026-05-01")
print(answer.text)                       # The operating margin was 7.2% in the half year.
print(answer.citations[0].doc, answer.verified)
```

For production quality, pass model names: `dense="BAAI/bge-m3"`, `reranker="hanaml/rerank-m"`,
`model="Qwen/Qwen3-32B"` behind an OpenAI-compatible endpoint (`OPENAI_BASE_URL`, e.g. vLLM), and give
`cite_check.Checker` an NLI model and the `hanaml/llama-cite-verify-3b` verifier. Each package README lists its
options.

## Architecture

```mermaid
flowchart LR
    docs[(Documents)] --> ingest[Split at headings,<br/>then window]
    ingest --> bm25[BM25 index<br/>Tantivy]
    ingest --> dense[Dense index<br/>bge-m3]
    user([User turns]) --> raw[(Raw turns)]
    raw --> plan[Budgeted cycle<br/>TIDE knapsack]
    plan -->|graph write| graph[(Bitemporal graph<br/>graph-memory)]

    q([Question]) --> recall[Recall facts valid<br/>at question time]
    graph --> recall
    raw --> recall
    recall --> rewrite[Resolve 'my employer',<br/>'where I live']
    rewrite --> fuse[RRF fusion]
    bm25 --> fuse
    dense --> fuse
    fuse --> rerank[Cross-encoder rerank]
    rerank --> loop[Answer loop<br/>search, open_section,<br/>follow_reference, answer]
    loop --> verify[cite-check<br/>claims, support, figures]
    verify -->|issues| revise[Revise flagged claims]
    revise --> verify
    verify --> out([Answer with citations])
```

How the pieces connect:

- `hana_knowledge.KnowledgeBase` asks `graph-memory` for the facts valid at the time of the question and rewrites
  the question with them before retrieval ("my employer" in February is not "my employer" in May).
- Chunks never cross a heading and keep a section path (`ARTICLE 7 COVENANTS / 7.11 Financial covenants`), so the
  answer loop can open a section or follow a cross-reference by its number; a reference that no heading carries
  becomes a search, where BM25's exact matching of clause numbers helps.
- `cite-check` finds the passage each claim cites with `retrieval-kit`'s index and retriever, so the answering
  side and the checking side read documents the same way. Citations are `retrieval_kit.Citation` objects
  (document, page, character span), rendered as `[doc, p. N]` markers, the format `cite-check` reads.
- With `Memory(consolidation="budgeted")`, turns wait as raw text that recall still reads; `consolidate()` spends
  at most `budget` times the cost of writing every turn, a turn that supersedes a held belief first.
  `Memory.delete()` replays the remaining turns, and `Recall.epoch` marks what was built before a deletion.

## Results

### Reproduced by this repo

Offline, with the stand-ins above, so these check the mechanics, not model quality:

- `uv run python examples/end_to_end.py`: three of four questions are answered, each cited to a character span
  and verified by cite-check; "my employer" resolves to Northwind Traders on 2026-02-01 and to Harbour Logistics on
  2026-05-01; the question that no passage supports is declined.
- `uv run python examples/budgeted_memory.py`: of 9 turns over 4 cycles at `budget=0.4`, 5 are written to the
  graph for 0.039k extraction tokens, against 0.088k to write every turn as it arrives; the move and the job
  change are written in the cycle they arrive in; one old small-talk turn leaves recall; deleting the move
  reopens Manchester as the current address (epoch 1).
- `make test-python`: cite-check catches all 7 errors injected into
  `packages/cite-check/data/injected-errors.jsonl` (3 unit, 2 numeric, 1 unsupported, 1 contradicted) and raises
  nothing on the 3 clean reports.

### Reported in the papers

The project's three Papers report these numbers, with the models named in each setting; each row quotes its paper.
LEDGER and InForce are seeded simulation studies calibrated to published baselines, as their limitations sections
say. LEDGER's reconciliation, InForce's in-force closure and defeater search, and TIDE's hazard model are in the paper
repos.

| Metric | Baseline | Result | Setting | Paper |
| --- | --- | --- | --- | --- |
| BiTempQA accuracy, in-order ingest | 60.2% | 77.2% | 2,400 bitemporal questions, GPT-4o-mini reader, 5 seeds; Graphiti re-run vs. LEDGER | [LEDGER](https://hanaml.com/research/papers/temporal-graphrag-memory) |
| As-of questions, BiTempQA | 46.5% | 76.9% | What the store believed at a past transaction time; Graphiti re-run vs. LEDGER | LEDGER |
| Accuracy change under shuffled ingest | −11.1 pts | −0.4 pts | BiTempQA, sessions imported out of order; Graphiti re-run vs. LEDGER (write-time invalidation on LEDGER's extraction: −12.3) | LEDGER |
| LongMemEval-S accuracy | 58.2% | 69.4% | Qwen3-14B reader; the full history in a 128k context vs. LEDGER (Graphiti re-run 65.0%) | LEDGER |
| Temporal-reasoning and knowledge-update questions, LongMemEval-S | 41.4% and 52.6% | 63.2% and 74.4% | The same Qwen3-14B runs | LEDGER |
| Median context tokens per question | 113k | 4.1k | LongMemEval-S, Qwen3-14B reader; the full history vs. LEDGER | LEDGER |
| Stale fact returned after a correction | 23.1% | 3.0% | 400 scripted correction histories; a turn-level vector store vs. LEDGER | LEDGER |
| In-force citation precision | 0.735 | 0.939 | AmendCite test, 800 questions, Qwen3-32B generator; local NLI + numeric verifier vs. InForce | [InForce](https://hanaml.com/research/papers/agentic-rag-citation-verification) |
| Defeated citations accepted | 84% | 10% | DefeatProbe, 1,200 defeated citations; the local NLI verifier (MiniCheck-7B: 86%) vs. InForce | InForce |
| Answer accuracy, AmendCite | 61.0% | 73.3% | Same generator, controller and repair loop; verification against the cited spans vs. the in-force closure | InForce |
| FinanceBench accuracy | 66.3% | 72.5% | 150-question open-source sample; local verification (dense RAG: 53.9%) vs. InForce; the 95% intervals overlap | InForce |
| Answers with an in-force defect at α = 0.10 | 38.6% | 10.0% | Conformal risk control, 75.6% of claims shipped; a local verifier calibrated on local labels vs. InForce on in-force labels | InForce |
| Questions answered confidently with the evidence removed | 15.2% | 2.8% | Dense RAG vs. InForce | InForce |
| p50 latency per question | 3.1 s | 8.3 s | AmendCite; one-pass dense RAG vs. InForce | InForce |
| LoCoMo J and LongMemEval-S at 40% of eager compute | 68.6 and 62.4% | 69.7 and 65.1% | GPT-4o-mini reader, budget B = 0.40; eager graph consolidation of the LEDGER store vs. TIDE | [TIDE](https://hanaml.com/research/papers/agent-memory-consolidation) |
| LoCoMo J and LongMemEval-S at equal compute | 68.6 and 62.4% | 71.4 and 66.2% | GPT-4o-mini reader, B = 1.00; eager graph consolidation vs. TIDE | TIDE |
| Accuracy lost from 10 to 320 sessions | 11.6 pts | 6.7 pts | Horizon streams built from LongMemEval haystacks; eager graph consolidation vs. TIDE | TIDE |

### Reported by the components

`cite-check` reports these in its README; no paper covers them.

| Metric | Baseline | Result | Note |
| --- | --- | --- | --- |
| Unsupported-claim recall, injected errors | 0.58 | 0.83 | Whole-answer LLM review vs. claim-level verifier |
| False alarms per 100 claims | 6.1 | 4.4 | Same reports, same reviewer model |
| Numeric-error detection | 49% | 92% | Unit swaps, wrong periods, arithmetic; typed checks in code |
| FActScore-style precision, generated reports | 0.71 | 0.84 | After one verify-and-revise pass |

## Papers

The project's Papers form one lineage: a bitemporal graph as memory, then claim-level checking of answers that
read it, then a scheduler that decides which turns are worth writing into that graph.

| Date | Paper | What this repo implements | PDF | Paper repo |
| --- | --- | --- | --- | --- |
| 2025-07-23 | [LEDGER: Order-Invariant Bitemporal Reconciliation for Knowledge-Graph Memory in Long-Horizon LLM Agents](https://hanaml.com/research/papers/temporal-graphrag-memory) | `graph-memory`: valid and recorded time on every edge, corrections that retract instead of delete, valid-time recall, communities. It closes a contradicted edge at write time, which is fast but order-dependent; LEDGER's order-invariant reconciliation and BiTempQA are in the paper repo | [PDF](https://hanaml.com/papers/temporal-graphrag-memory.pdf) | [temporal-graphrag-memory](https://github.com/hanaml-lab/temporal-graphrag-memory) |
| 2026-07-08 | [InForce: Defeater-Aware Claim Verification for Agentic Retrieval over Amended Financial and Legal Filings](https://hanaml.com/research/papers/agentic-rag-citation-verification) | `retrieval-kit`'s answer loop and `cite-check`: atomic claims, the NLI and verifier cascade, figures recomputed in code, revision; the provision graph, the in-force closure, the defeater search and the conformal threshold are in the paper repo | [PDF](https://hanaml.com/papers/agentic-rag-citation-verification.pdf) | [agentic-rag-citation-verification](https://github.com/hanaml-lab/agentic-rag-citation-verification) |
| 2026-08-19 | [TIDE: Hazard-Scheduled, Budget-Constrained Memory Consolidation for Long-Horizon LLM Agents](https://hanaml.com/research/papers/agent-memory-consolidation) | `graph_memory.consolidate`: the value of consolidation and the greedy multiple-choice knapsack with the best-single-upgrade check, wired into `Memory`; the hazard model, the LoRA tier and the experiments are in the paper repo | [PDF](https://hanaml.com/papers/agent-memory-consolidation.pdf) | [agent-memory-consolidation](https://github.com/hanaml-lab/agent-memory-consolidation) |

## Models

- [rerank-m](https://hanaml.com/models/rerank-m): a 0.5B cross-encoder (from Qwen2.5-0.5B) that reorders retrieved
  passages; `Retriever(reranker="hanaml/rerank-m")`.
- [llama-cite-verify-3b](https://hanaml.com/models/llama-cite-verify-3b): a 3B verifier (from Llama 3.2 3B
  Instruct) that judges whether a cited passage supports a claim; cite-check's second tier. Built with Llama.

## Development

```bash
make test       # the Python packages, the integration tests, the Rust extension and the TypeScript reader
make lint       # ruff
make example    # both examples
```

Each package can be tested on its own: `cd packages/<name> && uv run pytest`. CI runs lint, the Python tests on
3.10 and 3.12, both examples, clippy and the Rust tests with the Python bindings, and the TypeScript tests on every
push to `main` and every pull request.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use this software, cite it as below (also in [CITATION.cff](CITATION.cff)), and cite the Paper whose
method you use.

```bibtex
@software{hanaml_hana_knowledge_2026,
  author  = {{hanaML}},
  title   = {hana-knowledge: grounded answers over documents with bitemporal memory and claim-level verification},
  year    = {2026},
  version = {0.6.0},
  url     = {https://github.com/hanaml-lab/hana-knowledge}
}

@techreport{agarwal2025ledger,
  author      = {Agarwal, Manav},
  title       = {LEDGER: Order-Invariant Bitemporal Reconciliation for Knowledge-Graph Memory in Long-Horizon LLM Agents},
  institution = {hanaML},
  type        = {Preprint},
  year        = {2025},
  url         = {https://hanaml.com/research/papers/temporal-graphrag-memory}
}

@techreport{agarwal2026claimlevel,
  author      = {Agarwal, Manav},
  title       = {InForce: Defeater-Aware Claim Verification for Agentic Retrieval over Amended Financial and Legal Filings},
  institution = {hanaML},
  type        = {Technical report},
  year        = {2026},
  url         = {https://hanaml.com/research/papers/agentic-rag-citation-verification}
}

@techreport{agarwal2026tide,
  author      = {Agarwal, Manav},
  title       = {{TIDE}: Hazard-Scheduled, Budget-Constrained Memory Consolidation for Long-Horizon {LLM} Agents},
  institution = {hanaML},
  type        = {Technical report},
  year        = {2026},
  url         = {https://hanaml.com/research/papers/agent-memory-consolidation}
}
```

## Licence

Apache-2.0, except `packages/graph-memory` and `packages/notes-reader`, which are MIT. See the `LICENSE` file in
each.
