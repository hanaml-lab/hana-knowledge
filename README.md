# hana-knowledge

Grounded answers over documents. A question is answered only from passages in your documents, every sentence of
the answer points at the characters it rests on, and the answer is checked claim by claim before it is returned.
When nothing supports an answer, the system says so.

hana-knowledge is three components that call each other, plus a thin pipeline that wires them together:

| Component | What it does |
| --- | --- |
| [`retrieval-kit`](packages/retrieval-kit) | Hybrid BM25 and dense retrieval, reciprocal-rank fusion, cross-encoder reranking, and an answer loop that cites a character span for every claim or abstains. |
| [`graph-memory`](packages/graph-memory) | Long-term memory for each user as a bitemporal knowledge graph: when a fact held, and when the system learned it. |
| [`cite-check`](packages/cite-check) | Splits an answer or report into claims, finds the cited passage for each, checks support and recomputes every figure in code. |
| [`hana-knowledge`](packages/hana-knowledge) | The pipeline: ingest, index and graph, retrieve, answer, verify. |
| [`notes-reader`](packages/notes-reader) | TypeScript reader for graph-memory's dated-notes export, for apps that show what the agent remembers. |

## Architecture

```mermaid
flowchart LR
    docs[(Documents)] --> ingest[Ingest and chunk]
    ingest --> bm25[BM25 index<br/>Tantivy]
    ingest --> dense[Dense index<br/>bge-m3]
    user([User turns]) --> extract[Extract dated facts]
    extract --> graph[(Bitemporal graph<br/>graph-memory)]

    q([Question]) --> recall[Recall facts valid<br/>at question time]
    graph --> recall
    recall --> rewrite[Resolve 'my employer',<br/>'where I live']
    rewrite --> fuse[RRF fusion]
    bm25 --> fuse
    dense --> fuse
    fuse --> rerank[Cross-encoder rerank]
    rerank --> answer[Answer rounds<br/>cite span or abstain]
    answer --> verify[cite-check<br/>claims, support, figures]
    verify -->|issues| revise[Revise flagged claims]
    revise --> verify
    verify --> out([Answer with citations])
```

How the pieces connect:

- `cite-check` finds the passage each claim cites with `retrieval-kit`'s index and retriever, so the answer side and
  the checking side read documents the same way.
- `hana_knowledge.KnowledgeBase` asks `graph-memory` for the facts valid at the time of the question and rewrites the
  question with them before retrieval ("my employer" in February is not "my employer" in May).
- Citations are `retrieval_kit.Citation` objects (document, page, character span); the pipeline renders them as
  `[doc, p. N]` markers, which is the format `cite-check` reads.
