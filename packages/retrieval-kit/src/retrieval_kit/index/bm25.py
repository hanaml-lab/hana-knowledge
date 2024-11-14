"""Okapi BM25.

The Tantivy extension in ``tantivy-ext/`` is used when it is installed; this module is the
reference implementation and the fallback, and both rank with the same formula.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens. Keeps dotted and hyphenated terms such as clause numbers whole."""
    return [t.lower() for t in _WORD.findall(text)]


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def score(self, query: str, doc_id: int) -> float:
        tf, dl = self.tfs[doc_id], self.lengths[doc_id]
        total = 0.0
        for t in tokenize(query):
            f = tf.get(t, 0)
            if f:
                norm = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                total += self.idf[t] * f * (self.k1 + 1) / norm
        return total

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        candidates = {d for t in set(tokenize(query)) for d in self.postings.get(t, ())}
        scored = [(d, self.score(query, d)) for d in candidates]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
