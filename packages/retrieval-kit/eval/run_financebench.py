"""FinanceBench open-source sample: evidence-page recall@10 and answer accuracy.

    python eval/run_financebench.py --config configs/financebench.yaml --pdfs pdfs/

The question file is downloaded from the FinanceBench repository on first use. The filings
themselves (``--pdfs``) come from the same repository's ``pdfs/`` folder. Answers are graded by
normalised numeric or string match; the README's accuracy figures were human-checked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

from retrieval_kit import Answerer, Config, Index, Retriever

QUESTIONS_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
)


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(QUESTIONS_URL, path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"-?\d[\d,]*\.?\d*", text)}


def correct(predicted: str, gold: str) -> bool:
    gold_nums = numbers(gold)
    if gold_nums:
        return bool(gold_nums & numbers(predicted))
    return gold.strip().lower() in predicted.lower()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pdfs", default="pdfs/")
    ap.add_argument("--questions", default="data/financebench_open_source.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    questions = load_questions(Path(args.questions))[: args.limit]
    index = Index.build(args.pdfs, chunk_tokens=cfg.chunk_tokens, overlap=cfg.overlap)
    retriever = Retriever(index, dense=cfg.dense, reranker=cfg.reranker, k=10,
                          rrf_k=cfg.rrf_k, rerank_top=cfg.rerank_top)
    answerer = Answerer(retriever, model=cfg.model, max_rounds=cfg.max_rounds,
                        answer_top=cfg.answer_top, abstain_threshold=cfg.abstain_threshold)

    recall_hits = accurate = abstained = 0
    for q in questions:
        gold_pages = {(e["doc_name"], e["evidence_page_num"] + 1) for e in q.get("evidence", [])}
        hits = retriever.search(q["question"], k=10)
        found = {(Path(h.chunk.doc).stem, h.chunk.page) for h in hits}
        recall_hits += bool(gold_pages & found)
        result = answerer.ask(q["question"])
        abstained += result.abstained
        accurate += (not result.abstained) and correct(result.text, q["answer"])
    n = len(questions) or 1
    print(json.dumps({
        "questions": len(questions),
        "recall@10": round(recall_hits / n, 3),
        "accuracy": round(accurate / n, 3),
        "abstained": round(abstained / n, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
