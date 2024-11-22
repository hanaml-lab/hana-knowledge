"""Reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    """Fuse ranked lists by summing 1 / (k + rank), ranks starting at 1.

    Items are compared by identity only, so the lists' raw score scales never matter. Ties are
    broken by the best rank an item reached in any list, then by first appearance.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[Hashable, float] = {}
    best: dict[Hashable, int] = {}
    order: dict[Hashable, int] = {}
    for ranking in rankings:
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # a duplicate within one list counts once, at its best rank
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, rank), rank)
            order.setdefault(item, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], order[kv[0]]))
