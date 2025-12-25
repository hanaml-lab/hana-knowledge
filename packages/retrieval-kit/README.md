# retrieval-kit

Part of [hana-knowledge](../../README.md). Used by `cite-check` to find the passage a claim cites, and by the
`hana_knowledge` pipeline to answer.

Hybrid retrieval for question answering over long documents. A question goes through BM25 and dense retrieval, the
two lists are fused, a cross-encoder reranks the top of the fused list, and an agent decides whether it has enough
evidence to answer. Every sentence of the answer cites the passage it rests on; when no passage supports an answer,
the agent says so instead of guessing.

## Install

```bash
# from the root of hana-knowledge
pip install "./packages/retrieval-kit[gpu]"    # or [cpu] for ONNX on CPU
pip install "./packages/retrieval-kit[pdf]"    # Docling, to index PDFs
# optional: the BM25 index as a small Rust extension over Tantivy (needs a Rust toolchain)
pip install ./packages/retrieval-kit/tantivy-ext
```

Python 3.10 or later. Without the extension, BM25 runs in pure Python with the same formula. Without the model
extras, a hashing encoder and a term-overlap reranker stand in for bge-m3 and the cross-encoder, which is what the
test suite uses.

## Quickstart

```python
from retrieval_kit import Index, Retriever, Answerer

index = Index.build("filings/", chunk_tokens=384, overlap=64)   # PDFs are parsed with Docling
retriever = Retriever(index, dense="BAAI/bge-m3", reranker="hanaml/rerank-m", k=10)

answerer = Answerer(retriever, model="Qwen/Qwen3-32B", max_rounds=3)
result = answerer.ask("What was the operating margin in fiscal 2023?")

print(result.text)
for c in result.citations:
    print(c.doc, c.page, c.span)
```

`result.abstained` is true when the agent found no passage that supports an answer.

The generator is called through any OpenAI-compatible endpoint, set with `OPENAI_BASE_URL` (default
`http://localhost:8000/v1`, e.g. `vllm serve Qwen/Qwen3-32B`). Pass `model=None` for a small extractive answerer
that needs no LLM.

## How it works

1. **Two indexes.** BM25 (Tantivy) finds exact terms such as ticker symbols and clause numbers; bge-m3 finds paraphrases.
2. **Fusion.** Reciprocal-rank fusion (`k=60`) merges the two lists without tuning score scales.
3. **Rerank.** A cross-encoder scores the top 50 fused passages against the question.
4. **Agent rounds.** The model reads the top passages, then either answers, writes a follow-up query, or abstains. It stops after `max_rounds` or after a round that brings no new page.
5. **Citations.** Each claim in the answer is tied to a character span. `cite.py` checks the span exists in the passage before the answer is returned.

## Configuration

```yaml
# configs/financebench.yaml
chunk_tokens: 384
fusion: rrf
rrf_k: 60
rerank_top: 50
answer_top: 8
max_rounds: 3
abstain_threshold: 0.35   # reranker score below which a passage is not treated as evidence
dense: BAAI/bge-m3
reranker: hanaml/rerank-m
model: Qwen/Qwen3-32B
```

## Benchmarks

FinanceBench, 150-question public sample, Qwen3-32B as the generator in both columns. Run from
`packages/retrieval-kit` with `python eval/run_financebench.py --config configs/financebench.yaml --pdfs pdfs/`, where `pdfs/` holds the filings
from the FinanceBench repository; the question file is downloaded on first run.

| Metric | Dense only (bge-m3) | retrieval-kit |
| --- | --- | --- |
| Recall@10, evidence pages | 0.61 | 0.79 |
| Answer accuracy, open-book | 54% | 71% |
| Citation precision (claim-level, human-graded) | 0.68 | 0.88 |
| Unsupported answers instead of abstaining | 17% | 6% |
| p50 latency per question, one H100 | 3.1 s | 6.8 s |

The extra retrieval rounds roughly double latency. Set `max_rounds: 1` if that matters more than recall.

## Development

```bash
cd packages/retrieval-kit
pytest
cargo test --manifest-path tantivy-ext/Cargo.toml
```

## Licence

Apache-2.0. See `LICENSE`.
