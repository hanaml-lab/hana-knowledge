# cite-check

Part of [hana-knowledge](../../README.md). The pipeline runs it on every answer before returning it.

Asking a model to review a whole report finds some errors and misses most numeric ones. cite-check reviews claim
by claim. It splits a report into atomic claims, finds the passages each claim cites, checks support with an NLI
model and then a stronger reviewer, and recomputes every figure (totals, growth rates, unit conversions) in code
rather than trusting a model's arithmetic. The lab's `llama-cite-verify-3b` is trained for the support check.

## Install

```bash
# from the root of hana-knowledge
pip install ./packages/retrieval-kit "./packages/cite-check[models,sql]"
```

Without the `models` extra a lexical support check stands in for the NLI model, so the tool runs offline.

## Quickstart

```python
from cite_check import Checker

checker = Checker(
    nli="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    verifier="hanaml/llama-cite-verify-3b",
    reviewer="Qwen/Qwen3-32B",
)
review = checker.check(report="out/q3-summary.md", sources="sources/")

for issue in review.issues:
    print(issue.claim, issue.kind, issue.evidence)   # kind: unsupported | numeric | unit | contradicted

revised = checker.revise(review)    # rewrites only the claims that were flagged
```

The verifier and reviewer are called through an OpenAI-compatible endpoint (`OPENAI_BASE_URL`).

```bash
cite-check out/q3-summary.md --sources sources/ --format html > review.html
```

Citations in the report are written `[source]` or `[source, p. 3]`, where `source` names a file in the sources
folder. A source may be a CSV table, in which case figures are recomputed from it.

## How it works

1. **Claims.** The report is split into atomic claims, each tied to the citations around it.
2. **Retrieve.** For each claim, the cited source is searched (retrieval-kit, with bge-reranker-v2-m3 when installed) for the passage most likely to support it.
3. **Support.** An NLI model screens each claim–passage pair; uncertain pairs go to the verifier, and disagreements to the reviewer.
4. **Numbers.** Figures are parsed with their scale and unit (`units.py`). Where a source holds a table, the figure is recomputed, with DuckDB (`queries/figures.sql`) or the equivalent Python, and compared.
5. **Revise.** Flagged claims are rewritten from the evidence, and the new claims are checked once more.

## Configuration

| Setting | Default | What it does |
| --- | --- | --- |
| `nli_accept` | 0.9 | NLI entailment above which a claim is accepted without the verifier |
| `numeric_tolerance` | 0.005 (0.5%) | Relative difference allowed when recomputing a figure |
| `revise_passes` | 1 | Verify-and-revise rounds |

## Benchmarks

Reports with injected errors (wrong figures, swapped units, reversed conditions). Same reviewer model in both columns.

| Metric | Whole-answer review | cite-check |
| --- | --- | --- |
| Unsupported-claim recall | 0.58 | 0.83 |
| False alarms per 100 claims | 6.1 | 4.4 |
| Numeric-error detection | 49% | 92% |
| FActScore-style precision after one revise pass | 0.71 | 0.84 |

`data/injected-errors.jsonl` holds a small public sample of the injected-error format; the test suite checks
that every injected error in it is caught.

## Licence

Apache-2.0.
