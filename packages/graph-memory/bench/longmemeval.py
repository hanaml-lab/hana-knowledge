"""LongMemEval runner: python bench/longmemeval.py --split s

Each question's chat history is written to a fresh Memory turn by turn, with the session dates.
The recalled context goes to a reader model, and a judge model grades the answer against the
reference, following the LongMemEval protocol. Both are called through an OpenAI-compatible
endpoint (``OPENAI_BASE_URL``). Seeds are fixed so a rerun samples the same questions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from graph_memory import Memory

DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_{split}.json"


def chat(model: str, prompt: str) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
    })
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


def parse_date(text: str):
    return datetime.strptime(text.split(" (")[0], "%Y/%m/%d").date()


def load(split: str, cache: Path) -> list[dict]:
    path = cache / f"longmemeval_{split}.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(split=split), path)
    return json.loads(path.read_text(encoding="utf-8"))


def run(item: dict, reader: str, judge: str, extractor: str) -> bool:
    memory = Memory(extractor=extractor)
    for session, when in zip(item["haystack_sessions"], item["haystack_dates"], strict=True):
        at = parse_date(when)
        for turn in session:
            if turn["role"] == "user":
                memory.add(user="user", text=turn["content"], at=at)
    recalled = memory.recall(user="user", query=item["question"], at=parse_date(item["question_date"]))
    answer = chat(reader, f"Memory:\n{recalled.context}\n\nToday is {item['question_date']}.\n"
                          f"Question: {item['question']}\nAnswer briefly.")
    verdict = chat(judge, f"Question: {item['question']}\nReference answer: {item['answer']}\n"
                          f"Model answer: {answer}\nIs the model answer correct? Reply yes or no.")
    return verdict.lower().startswith("yes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["s", "m", "oracle"], default="s")
    ap.add_argument("--reader", default="Qwen/Qwen3-32B")
    ap.add_argument("--judge", default="Qwen/Qwen3-32B")
    ap.add_argument("--extractor", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=".cache/longmemeval")
    args = ap.parse_args(argv)

    items = load(args.split, Path(args.cache))
    random.Random(args.seed).shuffle(items)
    items = items[: args.limit] if args.limit else items
    by_type: dict[str, list[bool]] = defaultdict(list)
    for item in items:
        by_type[item["question_type"]].append(run(item, args.reader, args.judge, args.extractor))
    every = [ok for oks in by_type.values() for ok in oks]
    report = {"accuracy": sum(every) / max(len(every), 1), "n": len(every)}
    report.update({t: round(sum(v) / len(v), 3) for t, v in sorted(by_type.items())})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
